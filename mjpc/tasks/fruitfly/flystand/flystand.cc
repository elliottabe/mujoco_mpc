// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/fruitfly/flystand/flystand.h"

#include <mujoco/mujoco.h>

#include <string>

#include "mjpc/utilities.h"

namespace mjpc::fruitfly {

std::string FlyStand::XmlPath() const {
  return GetModelPath("fruitfly/flystand/task.xml");
}
std::string FlyStand::Name() const { return "Fruitfly Stand"; }

// ------------------ Residuals for fruitfly stand task ------------
//   Number of residuals: 6
//     Residual (0): Desired height
//     Residual (1): Balance: COM_xy - average(feet position)_xy
//     Residual (2): Com Vel: should be 0 and equal feet average vel
//     Residual (3): Control: minimise control
//     Residual (4): Joint vel: minimise joint velocity
//   Number of parameters: 1
//     Parameter (0): height_goal
// ----------------------------------------------------------------
void FlyStand::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;

  // ----- Height: head feet vertical error ----- //

  // feet sensor positions
  double* f1_position = SensorByName(model, data, "T1_left_pos");
  double* f2_position = SensorByName(model, data, "T1_right_pos");
  double* f3_position = SensorByName(model, data, "T2_left_pos");
  double* f4_position = SensorByName(model, data, "T2_right_pos");
  double* f5_position = SensorByName(model, data, "T3_left_pos");
  double* f6_position = SensorByName(model, data, "T3_right_pos");
  double* head_position = SensorByName(model, data, "thorax_pos");
  double head_feet_error = head_position[2];
  residual[counter++] = head_feet_error - parameters_[0];

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double* com_position = SensorByName(model, data, "thorax_subtreecom");
  double* com_velocity = SensorByName(model, data, "thorax_subtreelinvel");
  double kFallTime = 0.2;
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  mju_addToScl3(capture_point, com_velocity, kFallTime);

  // average feet xy position
  double fxy_avg[2] = {0.0};
  mju_addTo(fxy_avg, f1_position, 2);
  mju_addTo(fxy_avg, f2_position, 2);
  mju_addTo(fxy_avg, f3_position, 2);
  mju_addTo(fxy_avg, f4_position, 2);
  mju_addTo(fxy_avg, f5_position, 2);
  mju_addTo(fxy_avg, f6_position, 2);
  mju_scl(fxy_avg, fxy_avg, 0.25, 2);

  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel+6, model->nv-6);
  counter += model->nv-6;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

}  // namespace mjpc::fruitfly
