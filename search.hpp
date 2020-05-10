#pragma once

#include <unistd.h>
#include <sys/time.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <list>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "net.hpp"
#include "reversi.hpp"


const double VL_COUNT = 4;
const double ROOT_NOISE_EPS = 0.25;
const double ROOT_NOISE_ALPHA = 0.1;
const double C_PUCT = 2.0;

const int BATCH_SIZE = 128;


// time counter
inline double current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

// propability distribution
inline std::vector<double> dirichret_random(std::mt19937& dice, int k, double alpha)
{
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> ans;

    double sum = 0;
    for (int i = 0; i < k; i++)
    {
        double r = gamma(dice);
        ans.push_back(r);
        sum += r;
    }

    for (double& v : ans) v /= sum;
    return ans;
}


template <typename T>
struct Atomic : public std::atomic<T>
{
    T operator +=(T src)
    {
        T expected = this->load();
        while (!std::atomic_compare_exchange_weak(this, &expected, T(expected + src)));
        return expected;
    }

    Atomic<T>& operator =(const Atomic<T>& src)
    {
        std::atomic<T>::store(src.load());
        return *this;
    }

    Atomic() {}
    Atomic(const T& src)
    {
        std::atomic<T>::store(src);
    }
    Atomic(const Atomic<T>& src)
    {
       std::atomic<T>::store(src.load());
    }
};


template <class node_t>
struct edge_t
{
    short a;
    Atomic<short> vl = 0;
    Atomic<float> p = -FLT_MAX;
    Atomic<node_t*> next = nullptr;

    ~edge_t()
    {
        while (vl.load() != 0) usleep(1e-2 * 1000000);
        if (next.load() != nullptr) delete next.load();
    }

    void set(int action)
    {
        a = action;
    }

    double next_qsum() const
    {
        node_t *node = next;
        if (node == nullptr) return 0;
        return -node->q_sum.load(); // expected situation: flip color by every turn
    }

    long long next_n() const
    {
        node_t *node = next;
        if (node == nullptr) return 0;
        return node->n.load();
    }

    bool regist_next_node(node_t *node)
    {
        node_t* null_node = nullptr;
        return next.compare_exchange_strong(null_node, node);
    }

    std::string to_string(const Board& b) const
    {
        std::ostringstream oss;
        oss << std::fixed;
        oss << "q " << std::setprecision(4) << std::setw(7) << next_qsum() / (next_n() + 1e-4);
        oss << " p " << std::setw(6) << float(p);
        node_t *node = next.load();
        if (node != nullptr)
        {
            auto pv = node->pv();
            oss << " pv " << b.action2string(a);
            for (int na : pv) oss << " " << b.action2string(na);
        }
        return oss.str();
    }
};


struct Node;
using Edge = edge_t<Node>;

struct Node
{
    Node* prev = nullptr;
    int prev_idx = -1;
    Atomic<float> v = -FLT_MAX;
    Atomic<double> q_sum = 0;
    Atomic<long long> n = 0;

    std::vector<Edge> edges;

    Node(std::vector<int> actions, Node *node = nullptr, int idx = -1)
    {
        clear(actions, node, idx);
    }

    void clear(const std::vector<int>& actions, Node *node, int idx)
    {
        prev = node;
        prev_idx = idx;
        edges.resize(actions.size());
        for (int i = 0; i < int(actions.size()); i++) edges[i].set(actions[i]);
    }

    void set(const std::vector<float>& policy, float value)
    {
        double p_sum = 0;
        for (Edge& e : edges) p_sum += policy[e.a];
        for (Edge& e : edges) e.p = policy[e.a] / (p_sum + 1e-16);
        v = value;
    }

    float update(float value)
    {
        q_sum += value;
        n += 1;
        return value;
    }

    int best_action_index() const
    {
        int best_idx = -1;
        double best_n = 0;
        for (int i = 0; i < int(edges.size()); i++)
        {
            double n = edges[i].next_n();
            if (n > best_n)
            {
                best_idx = i;
                best_n = n;
            }
        }
        return best_idx;
    }

    std::list<int> pv() const
    {
        std::list<int> seq;
        int idx = best_action_index();
        if (idx >= 0)
        {
            seq.push_back(edges[idx].a);
            seq.splice(seq.end(), edges[idx].next.load()->pv());
        }
        return seq;
    }

    std::string to_string(const Board& b) const
    {
        std::ostringstream oss;
        int idx = best_action_index();
        if (idx >= 0)
        {
            const Edge& e = edges[idx];
            oss << std::fixed;
            oss << "best " << b.action2string(e.a);
            oss << " n " << std::setw(5) << int(e.next_n()) << "/" << std::setw(5) << int(n);
            oss << " " << e.to_string(b);
        }
        return oss.str();
    }
};

struct TreeSearch
{
    Node *root = nullptr;
    Node *prev_root = nullptr;
    mutable std::mutex search_lock;
    std::atomic<bool> end_flag;

    // temporal values
    double start_time, limit_time;

    // for async network computation
    std::queue<std::pair<Node*, std::vector<float>>> in_queue;
    std::queue<std::pair<Node*, std::array<std::vector<float>, 2>>> out_queue;
    mutable std::mutex in_lock, out_lock;

    // configuration
    int num_threads = 0;


    TreeSearch(const Net& net, int search_threads = 1, int gpu_threads = 1)
    {
        num_threads = search_threads;

        // neural network threads
        std::vector<std::thread> threads;
        for (int i = 0; i < gpu_threads; i++)
        {
            std::thread t(&TreeSearch::net_thread, this, i, &net);
            t.detach();
        }
    }

    ~TreeSearch()
    {
        stop();
    }

    void stop()
    {
        end_flag = true;
        // wait until all threads end
        search_lock.lock();
        search_lock.unlock();
    }

    int go(const Board& b, double time)
    {
        std::cerr << "thinking..." << std::endl;
        start_time = current_time();
        limit_time = time;
        return think(&b);
    }

    void ponder(const Board& b)
    {
        // open ponder thread
        std::cerr << "pondering..." << std::endl;
        start_time = current_time();
        limit_time = 1000000;
        std::thread t(&TreeSearch::think, this, &b);
        t.detach();
    }

    void next(int action)
    {
        if (root == nullptr) return;
        prev_root = root;

        for (Edge& e : root->edges)
        {
            if (e.a == action)
            {
                root = e.next;
                root->prev = nullptr;
                return;
            }
        }
        root = nullptr;
    }

    // inner function
    int think(const Board *b)
    {
        search_lock.lock();
        end_flag = false;

        if (root == nullptr)
        {
            root = new Node(b->legal_actions());
            request(root, *b);
        }

        // timer thread
        std::thread timer(&TreeSearch::timer_thread, this, b);
        timer.detach();

        // gabage collection thread
        if (prev_root != nullptr)
        {
            std::thread gc(&TreeSearch::gc_thread, this, prev_root, root->prev_idx);
            gc.detach();
        }

        // search threads
        if (num_threads > 1)
        {
            std::vector<std::thread> threads;

            for (int i = 0; i < num_threads; i++)
            {
                threads.push_back(std::thread(&TreeSearch::search_thread, this, i, b));
            }
            for (auto& t : threads) t.join();
        }
        else
        {
            TreeSearch::search_thread(0, b);
        }

        int best_idx = root->best_action_index();
        int best = root->edges[best_idx].a;

        std::cerr << summary(*b);
        std::cerr << "best action " << b->action2string(best) << std::endl;

        search_lock.unlock();
        return best;
    }

    void search_thread(int thread_index, const Board* board)
    {
        Board b = *board;
        std::mt19937 dice(thread_index);

        while (!end_flag.load())
        {
            check_outputs();
            if (check_inputs() >= 2048) usleep(0.1 * 1000000);

            Board bb = b;
            search(bb, root, dice, 0);
        }
    }

    std::string summary(const Board& b) const
    {
        std::vector<std::pair<long long, int>> ns;
        for (int i = 0; i < int(root->edges.size()); i++)
        {
            ns.push_back(std::make_pair(-root->edges[i].next_n(), i));
        }
        std::sort(ns.begin(), ns.end());
        int digits = ns.size() ? std::to_string(-ns[0].first).length() : 0;

        std::ostringstream oss;
        oss << std::fixed << "summary" << std::endl;

        for (int i = 0; i < std::min(8, int(ns.size())); i++)
        {
            int idx = ns[i].second;
            auto& e = root->edges[idx];
            oss << i + 1 << "." << b.action2string(e.a);
            oss << " n " << std::setw(digits) << e.next_n() << " " << e.to_string(b) << std::endl;
        }
        return oss.str();
    }

    float search(Board& b, Node* node, std::mt19937& dice, int depth)
    {
        if (b.terminal()) return node->update(b.reward(b.color));

        // wait until network outputs returns
        while (node->v.load() < -1) check_outputs();

        if (depth >= 256) return node->update(node->v);

        int idx = pucb_action_index(*node, dice, depth);
        Edge& edge = node->edges[idx];
        edge.vl += 1;
        b.play(edge.a);

        Node *next = edge.next;
        if (!next)
        {
            next = new Node(b.legal_actions(), node, idx);
            if (edge.regist_next_node(next))
            {
                if (!b.terminal())
                {
                    request(next, b);
                    return -FLT_MAX;
                }
            }
            else
            {
                // If another node had already been registered, we should delete created node.
                delete next;
                next = edge.next;
            }
        }

        // recursive search
        float v = search(b, next, dice, depth + 1);
        if (v >= -1)
        {
            v = -v;
            node->update(v);
            edge.vl -= 1;
        }

        return v;
    }

    int pucb_action_index(const Node& node, std::mt19937& dice, int depth)
    {
        std::vector<double> d;
        if (depth == 0) d = dirichret_random(dice, node.edges.size(), ROOT_NOISE_ALPHA);

        // choose best action
        int best_idx = -1;
        double best_ucb = -DBL_MAX;
        double v_base = (node.q_sum + node.v) / (node.n + 1);

        int acount = 0;
        for (int i = 0; i < int(node.edges.size()); i++)
        {
            const Edge& e = node.edges[i];

            double n = e.next_n();
            double vl = e.vl.load() * VL_COUNT;

            double q = n > 0 ? (e.next_qsum() - vl + v_base) / (n + vl + 1) : v_base;

            double p = e.p;
            if (depth == 0) p = p * (1 - ROOT_NOISE_EPS) + d[i] * ROOT_NOISE_EPS;

            // caluculate PUCB value
            double u = q + C_PUCT * p * std::sqrt(node.n + 1) / (n + 1);

            if (u > best_ucb)
            {
                best_idx = i;
                best_ucb = u;
            }
        }

        return best_idx;
    }

    // using neural nets
    void request(Node *node, const Board& b)
    {
        std::vector<float> fea = b.feature();
        in_lock.lock();
        in_queue.push(std::make_pair(node, std::move(fea)));
        in_lock.unlock();
    }

    int check_inputs() const
    {
        in_lock.lock();
        int count = in_queue.size();
        in_lock.unlock();
        return count;
    }

    void check_outputs()
    {
        std::vector<std::pair<Node*, std::array<std::vector<float>, 2>>> outputs;

        out_lock.lock();
        while (!out_queue.empty())
        {
            outputs.push_back(std::move(out_queue.front()));
            out_queue.pop();
        }
        out_lock.unlock();

        for (auto& npv : outputs)
        {
            Node *node = npv.first;
            std::vector<float>& p = npv.second[0];
            float v = npv.second[1][0];
            node->set(p, v);

            int idx = node->prev_idx;
            node = node->prev;

            // update upper nodes
            while (node != nullptr)
            {
                v = -v;
                node->update(v);
                node->edges[idx].vl -= 1;

                idx = node->prev_idx;
                node = node->prev;
            }
        }
    }

    void net_thread(int gpu_index, const Net *net)
    {
        // configuration of neural net and GPU
        const Net *gnet = net;
        // (...)

        while (true)
        {
            std::vector<Node*> leaves;
            std::vector<std::vector<float>> features;

            in_lock.lock();
            int in_count = in_queue.size();
            if (in_count == 0)
            {
                in_lock.unlock();
                usleep(0.1 * 1000000);
                continue;
            }

            for (int i = 0; i < std::min(BATCH_SIZE, in_count); i++)
            {
                auto& req = in_queue.front();
                leaves.push_back(req.first);
                features.push_back(std::move(req.second));
                in_queue.pop();
            }
            in_lock.unlock();

            // calculate neural network
            auto pv = gnet->evaluate(features);

            out_lock.lock();
            for (int i = 0; i < int(pv.size()); i++)
            {
                out_queue.push(std::make_pair(leaves[i], std::move(pv[i])));
            }
            out_lock.unlock();
        }
    }

    void timer_thread(const Board* b)
    {
        double prev_time = -1;
        while (!end_flag.load())
        {
            usleep(1e-2 * 1000000);
            double used_time = current_time() - start_time;
            if (used_time >= limit_time) end_flag = true;

            if (int(used_time) > int(prev_time))
            {
                prev_time = used_time;
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << std::setw(5) << used_time << " sec. ";
                std::cerr << oss.str() + root->to_string(*b) << std::endl;
            }
        }
    }

    void gc_thread(Node *prev_root, int prev_idx)
    {
        if (prev_idx >= 0) prev_root->edges[prev_idx].next = nullptr;
        delete prev_root;
    }
};
