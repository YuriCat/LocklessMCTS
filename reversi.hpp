#pragma once

#include <array>
#include <cassert>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>


const int D2[8][2] = {
    {-1,  0}, { 0, -1}, { 0,  1}, { 1,  0},
    {-1, -1}, {-1,  1}, { 1, -1}, { 1,  1},
};


struct Board
{
    const int L;
    int color;
    std::array<int, 2> score;
    int prev1, prev2;
    std::vector<int> board;

    Board(int length = 8):
    L(length)
    {
        board.resize(L * L);
        clear();
    }

    static int opponent(int c)
    {
        return 1 - c;
    }

    std::vector<int> feature_size() const
    {
        return {L, L, 2};
    }

    int action_size() const
    {
        return L * L + 1;
    }

    bool onboard(int x, int y) const
    {
        return 0 <= x && x < L && 0 <= y && y < L;
    }

    bool onboard(int pos) const
    {
        return 0 <= pos && pos < L * L;
    }

    char c2char(int c) const
    {
        if (c == 0) return 'X';
        if (c == 1) return 'O';
        return '.';
    }

    char x2char(int x) const
    {
        return 'A' + x;
    }

    char y2char(int y) const
    {
        return '1' + y;
    }

    std::string action2string(int action) const
    {
        if (action == L * L) return "PASS";
        int pos = action;
        std::ostringstream oss;
        oss << x2char(position2x(pos)) << y2char(position2y(pos));
        return oss.str();
    }

    std::string to_string() const
    {
        std::ostringstream oss;

        oss << "  ";
        for (int x = 0; x < L; x++) oss << x2char(x);
        oss << std::endl;

        for (int y = 0; y < L; y++)
        {
            oss << y2char(y) << " ";
            for (int x = 0; x < L; x++)
            {
                oss << c2char(board[xy2position(x, y)]);
            }
            oss << std::endl;
        }
        oss << " " << score[0] << " - " << score[1] << " " << c2char(color) << std::endl;

        return oss.str();
    }

    void clear()
    {
        fill(board.begin(), board.end(), -1);
        color = 0;

        // original state
        int mid = (L - 1) / 2;
        board[xy2position(mid, mid)] = 1;
        board[xy2position(mid, mid + 1)] = 0;
        board[xy2position(mid + 1, mid)] = 0;
        board[xy2position(mid + 1, mid + 1)] = 1;
        score.fill(2);

        prev1 = prev2 = -1;
    }

    void play(int action)
    {
        assert(legal(action));
        if (action != L * L)
        {
            int pos = action;
            auto counts = flip_counts(pos);
            int flipped = flip_stones(pos, counts);
            board[pos] = color;
            score[color] += 1 + flipped;
            score[opponent(color)] -= flipped;
        }
        color = opponent(color);
        prev2 = prev1;
        prev1 = action;
    }

    bool terminal() const
    {
        bool full = score[0] + score[1] == L * L;
        bool perfect = score[0] == 0 || score[1] == 0;
        bool pass2 = prev1 == L * L && prev2 == L * L;
        return full || perfect || pass2;
    }

    float reward(int c = 0) const
    {
        int s = count_score(c);
        if (s > 0) return 1;
        if (s < 0) return -1;
        return 0;
    }

    bool legal(int action) const
    {
        if (action == L * L)
        {
            // pass
            for (int i = 0; i < L * L; i++) if (flip_count(i) > 0) return false;
            return true;
        }
        else if (onboard(action))
        {
            if (flip_count(action) > 0) return true;
        }
        return false;
    }

    std::vector<int> legal_actions() const
    {
        std::vector<int> actions;
        for (int i = 0; i < L * L; i++)
        {
            if (legal(i)) actions.push_back(i);
        }
        if (actions.size() == 0)
        {
            actions.push_back(L * L); // pass
        }
        return actions;
    }

    std::vector<float> feature() const
    {
        std::vector<float> f(2 * L * L, 0);
        for (int pos = 0; pos < L * L; pos++)
        {
            if      (board[pos] == color)           f[pos] = 1;
            else if (board[pos] == opponent(color)) f[pos + L * L] = 1;
        }
        return f;
    }

    int count_score(int c) const
    {
        int diff = score[0] - score[1];
        return c != 0 ? -diff : diff;
    }

    int position2x(int pos) const
    {
        return pos % L;
    }

    int position2y(int pos) const
    {
        return pos / L;
    }

    int xy2position(int x, int y) const
    {
        return y * L + x;
    }

    int flip_count(int pos) const
    {
        auto counts = flip_counts(pos);
        return std::accumulate(counts.begin(), counts.end(), 0);
    }

    std::array<int, 8> flip_counts(int pos) const
    {
        std::array<int, 8> counts;
        counts.fill(0);

        if (!onboard(pos)) return counts;
        if (board[pos] != -1) return counts;
        int x_pos = position2x(pos);
        int y_pos = position2y(pos);

        for (int d = 0; d < 8; d++)
        {
            int flipped = 0;
            int x = x_pos;
            int y = y_pos;
            while (1)
            {
                x += D2[d][0];
                y += D2[d][1];
                int pos = xy2position(x, y);
                if (!onboard(x, y) || board[pos] == -1)
                {
                    flipped = 0; // no stones will be sandwiched
                    break;
                }
                if (board[pos] == color) break;
                flipped++;
            }
            counts[d] = flipped;
        }
        return counts;
    }

    int flip_stones(int action, const std::array<int, 8>& counts)
    {
        for (int d = 0; d < 8; d++)
        {
            int x = position2x(action);
            int y = position2y(action);
            for (int i = 0; i < counts[d]; i++)
            {
                x += D2[d][0];
                y += D2[d][1];
                assert(onboard(x, y));
                int pos = xy2position(x, y);
                board[pos] = opponent(board[pos]);
            }
        }
        return std::accumulate(counts.begin(), counts.end(), 0);
    }
};
