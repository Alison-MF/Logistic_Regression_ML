//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#ifndef STACK_PHY_LAYER_MACHINE_LEARNING_QLEARNING_H_
#define STACK_PHY_LAYER_MACHINE_LEARNING_QLEARNING_H_

#include <bits/stdc++.h> // header file for all c++ libraries
#include <fstream>
using namespace std;

struct Vetores_RL {

    vector<vector<double>> Reward_Matrix;
    vector<vector<double>> Q_Table;
    vector<double> cumulative_rewards_vector;
    int new_action[2];

    };

class Q_Learning {
public:
    Q_Learning();
    vector<vector<string>> load_file();
    void print_matrix(vector<vector<double>> m, int rows, int columns);
    int update_index_action(vector<vector<double>> Reward_Matrix, int state,int new_action[2]);
    double calc_max_qvalue(vector<vector<double>> Q_Table,vector<vector<double>> Reward_Matrix, int state);
    int predict_handover(vector<double> now_state, vector<vector<double>> Q_Table);
    void update_QTable(vector<double> init_state,struct Vetores_RL dataset);
    struct Vetores_RL training_phase(const std::vector<std::vector<double>> initial_states, int num_initial_states, struct Vetores_RL dataset);
    struct Vetores_RL get_reward_from_data(vector<vector<string>> record, struct Vetores_RL dataset);
    virtual ~Q_Learning();

private:
    double epsilon = 0.2;
    double cumulative_reward = 0.0;

public:
    int STATE_DIM = 8;
    int ACTION_NUM = 2;
    int HANDOVER = 1;
    int MAX_EPISODE = 1000;
    double alpha = 0.8;
    int MAX_ITERATIONS = 100;

};

#endif /* STACK_PHY_LAYER_MACHINE_LEARNING_QLEARNING_H_ */
