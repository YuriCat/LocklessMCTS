#pragma once

#include "unistd.h"
#include <array>
#include <cmath>
#include <random>
#include <vector>


// Since this is simple sample code, network only wait a few time and return uniform policy and zero value.

struct Net
{
    std::vector<int> feature_shape;
    int action_size;

    Net(std::vector<int> shape, int size)
    {
        feature_shape = shape;
        action_size = size;
    }

    std::vector<std::array<std::vector<float>, 2>> evaluate(std::vector<std::vector<float>> features) const
    {
        // make batch tensor from features and feature_shape
        // compute neural network

        double n = features.size();
        double k = std::sqrt(0.5);
        double m = n / 256 * k;
        double curve = m < k ? std::sqrt(m) : (m + std::sqrt(k) - k);
        double duration = 3e-1 * curve / std::sqrt(k);
        usleep(duration * 1000000);

        std::vector<std::array<std::vector<float>, 2>> out;

        // only returns random policy and value
        std::random_device rd;
        std::mt19937 dice(rd());
        std::uniform_real_distribution<float> uni(0, 1);

        for (auto& f : features)
        {
            std::vector<float> p;
            std::vector<float> v;

            for (int i = 0; i < action_size; i++) p.push_back(uni(dice));
            float sum = std::accumulate(p.begin(), p.end(), 0.0f);
            for (float& prob : p) prob /= sum;

            v.push_back(uni(dice) * 2 - 1);

            out.push_back({p, v});
        }

        return out;
    }
};