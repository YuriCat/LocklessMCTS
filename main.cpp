#include <iostream>
#include <random>

#include "search.hpp"


const int BOARD_SIZE = 6;

const int NUM_THREADS = 6;
const int NUM_GPUS = 2;
const double SEARCH_TIME = 10;

void selfplay()
{
    // self-play mode
    Board b(BOARD_SIZE);
    Net net(b.feature_size(), b.action_size());
    TreeSearch tree(net, NUM_THREADS, NUM_GPUS);

    while (!b.terminal())
    {
        std::cerr << b.to_string();
        int action = tree.go(b, SEARCH_TIME);

        b.play(action);
        tree.next(action);
    }
}

void battle()
{
    // battle mode against random agent
    Board b(BOARD_SIZE);
    Net net(b.feature_size(), b.action_size());
    TreeSearch tree(net, NUM_THREADS, NUM_GPUS);

    while (!b.terminal())
    {
        std::cerr << b.to_string();

        int action = -1;
        if (b.color == 0)
        {
            action = tree.go(b, SEARCH_TIME);
        }
        else
        {
            tree.ponder(b);
            usleep(SEARCH_TIME * 1000000);

            // random action
            auto actions = b.legal_actions();
            std::random_device rg;
            std::uniform_int_distribution<int> dice(0, actions.size() - 1);
            action = actions[dice(rg)];

            tree.stop();
        }

        b.play(action);
        tree.next(action);
    }
}

int main()
{
    //selfplay();
    battle();

    return 0;
}