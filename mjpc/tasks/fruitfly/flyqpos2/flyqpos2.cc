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

#include "mjpc/tasks/fruitfly/flyqpos2/flyqpos2.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <tuple>

#include "mjpc/utilities.h"

namespace {
// compute interpolation between mocap frames
std::tuple<int, int, double, double> ComputeInterpolationValues(double index,
                                                                int max_index) {
  int index_0 = std::floor(std::clamp(index, 0.0, (double)max_index));
  int index_1 = std::min(index_0 + 1, max_index);

  double weight_1 = std::clamp(index, 0.0, (double)max_index) - index_0;
  double weight_0 = 1.0 - weight_1;

  return {index_0, index_1, weight_0, weight_1};
}

constexpr double kFps = 50.0;

constexpr int kMotionLengths[] = {
    5800,   // FlyQpos2
    5800,   // FlyQpos2
    // 180,  // FlyStand
    // 1560,  // FlyQpos2
    // 8,  // FlyStand
};

// return length of motion trajectory
int MotionLength(int id) { return kMotionLengths[id]; }

// return starting keyframe index for motion
int MotionStartIndex(int id) {
  int start = 0;
  for (int i = 0; i < id; i++) {
    start += MotionLength(i);
  }
  return start;
}

// names for fruitfly bodies
const std::array<std::string, 30> body_names = {
    "coxa_T1_left",    "femur_T1_left",  "tibia_T1_left",   "tarsus_T1_left",
    "claw_T1_left",    "coxa_T1_right",  "femur_T1_right",  "tibia_T1_right",
    "tarsus_T1_right", "claw_T1_right",  "coxa_T2_left",    "femur_T2_left",
    "tibia_T2_left",   "tarsus_T2_left", "claw_T2_left",    "coxa_T2_right",
    "femur_T2_right",  "tibia_T2_right", "tarsus_T2_right", "claw_T2_right",
    "coxa_T3_left",    "femur_T3_left",  "tibia_T3_left",   "tarsus_T3_left",
    "claw_T3_left",    "coxa_T3_right",  "femur_T3_right",  "tibia_T3_right",
    "tarsus_T3_right", "claw_T3_right"};

// names for fruitfly bodies
const std::array<std::string, 36> joint_names = {
    "coxa_flexion_T1_left",  "coxa_twist_T1_left",   "femur_T1_left",  "femur_twist_T1_left",  "tibia_T1_left", "tarsus_T1_left", 
    "coxa_flexion_T1_right", "coxa_twist_T1_right", "femur_T1_right", "femur_twist_T1_right", "tibia_T1_right", "tarsus_T1_right", 
    "coxa_flexion_T2_left",  "coxa_twist_T2_left",   "femur_T2_left",  "femur_twist_T2_left",  "tibia_T2_left", "tarsus_T2_left", 
    "coxa_flexion_T2_right", "coxa_twist_T2_right", "femur_T2_right", "femur_twist_T2_right", "tibia_T2_right", "tarsus_T2_right",
    "coxa_flexion_T3_left",  "coxa_twist_T3_left",   "femur_T3_left",  "femur_twist_T3_left",  "tibia_T3_left", "tarsus_T3_left", 
    "coxa_flexion_T3_right", "coxa_twist_T3_right", "femur_T3_right", "femur_twist_T3_right", "tibia_T3_right", "tarsus_T3_right"};
}  // namespace
namespace mjpc::fruitfly {

std::string FlyQpos2::XmlPath() const {
  return GetModelPath("fruitfly/flyqpos2/task.xml");
}
std::string FlyQpos2::Name() const { return "Fruitfly Qpos2"; }

// ------------- Residuals for fruitfly tracking task -------------
//   Number of residuals:
//     Residual (0): Joint vel: minimise joint velocity
//     Residual (1): Control: minimise control
//     Residual (2): 
//     Residual (2-31): Tracking position: minimise tracking position error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//     Residual (31-66): Tracking velocity: minimise tracking velocity error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//   Number of parameters: 0
// ----------------------------------------------------------------
void FlyQpos2::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
  // ----- get mocap frames ----- //
  // get motion start index
  int start = MotionStartIndex(current_mode_);
  // get motion trajectory length
  int length = MotionLength(current_mode_);
  double current_index = (data->time - reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) = ComputeInterpolationValues(current_index, last_key_index);

  // ----- residual ----- //
  int counter = 0;

  // ----- joint velocity ----- //
  // for (ResidualFn::FlyJoint joint : ResidualFn::kJointAll)  {
  //   // current joint velocity
  //   residual[counter] = data->qvel[joint];
  //   counter += 1;
  // };

  // // ----- action ----- //
  // for (ResidualFn::FlyJoint joint : ResidualFn::kJointAll)  {
  //   // current joint velocity
  //   residual[counter] = data->ctrl[joint];
  //   counter += 1;
  // };
  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  
  // ----- position ----- //

  // Compute interpolated frame.
  auto get_body_mqpos = [&](const std::string &joint_name, double result[1]) {
    // std::string mocap_body_name = joint_name;
    int joint_body_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
    assert(0 <= joint_body_id);
    int body_mocapid = joint_body_id;
    assert(0 <= body_mocapid);

    // current frame
    mju_scl(
        result,
        model->key_qpos + model->nq * key_index_0 + body_mocapid,
        weight_0,1);

    // next frame
    mju_addToScl(
        result,
        model->key_qpos + model->nq * key_index_1 + body_mocapid,
        weight_1,1);
  };

  auto get_body_sensor_pos = [&](const std::string &joint_name,
                                 double result[1]) {
    std::string pos_sensor_name = "tracking_pos[" + joint_name + "]";
    double *sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
    mju_copy(result, sensor_pos, 1);
  };

  for (const auto &joint_name : joint_names) {
    double body_mpos[1];
    get_body_mqpos(joint_name, body_mpos);

    // current position
    double body_sensor_pos[1];
    get_body_sensor_pos(joint_name, body_sensor_pos);

    mju_sub(&residual[counter], body_mpos, body_sensor_pos,1);

    counter += 1;
  }

  // ----- velocity ----- //
  for (const auto &joint_name : joint_names) {
    std::string linvel_sensor_name = "tracking_vel[" + joint_name + "]";
    int joint_body_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
    assert(0 <= joint_body_id);
    int body_mocapid = joint_body_id;
    assert(0 <= body_mocapid);

    // compute finite-difference velocity
    mju_copy(&residual[counter], model->key_qpos + model->nq * key_index_1 + body_mocapid,1);
    mju_subFrom(&residual[counter], model->key_qpos + model->nq * key_index_0 + body_mocapid,1);
    mju_scl(&residual[counter], &residual[counter], kFps,1);

    // subtract current velocity
    double *sensor_linvel = SensorByName(model, data, linvel_sensor_name.c_str());
    mju_subFrom(&residual[counter], sensor_linvel,1);

    counter += 1;
  }

  CheckSensorDim(model, counter);
}

// --------------------- Transition for fruitfly task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void FlyQpos2::TransitionLocked(mjModel *model, mjData *d) {
  // get motion start index
  int start = MotionStartIndex(mode);
  // get motion trajectory length
  int length = MotionLength(mode);

  // check for motion switch
  if (residual_.current_mode_ != mode || d->time == 0.0) {
    residual_.current_mode_ = mode;       // set motion id
    residual_.reference_time_ = d->time;  // set reference time

    // set initial state
    mju_copy(d->qpos, model->key_qpos + model->nq * start, model->nq);
    mju_copy(d->qvel, model->key_qvel + model->nv * start, model->nv);
  }

  // indices
  double current_index = (d->time - residual_.reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  mj_markStack(d);

  mjtNum *mocap_pos_0 = mj_stackAllocNum(d, 3 * model->nmocap);
  mjtNum *mocap_pos_1 = mj_stackAllocNum(d, 3 * model->nmocap);

  // Compute interpolated frame.
  mju_scl(mocap_pos_0, model->key_mpos + model->nmocap * 3 * key_index_0,
          weight_0, model->nmocap * 3);

  mju_scl(mocap_pos_1, model->key_mpos + model->nmocap * 3 * key_index_1,
          weight_1, model->nmocap * 3);

  mju_copy(d->mocap_pos, mocap_pos_0, model->nmocap * 3);
  mju_addTo(d->mocap_pos, mocap_pos_1, model->nmocap * 3);

  mj_freeStack(d);
}

//  ============  task-state utilities  ============
// save task-related ids
void FlyQpos2::ResetLocked(const mjModel* model) {
  // ----------  task identifiers  ----------
  // residual_.jointVel_id_ = CostTermByName(model, "JointVel");
  residual_.control_id_ = CostTermByName(model, "Control");


  // ----------  model identifiers  ----------
  residual_.thorax_body_id_ = mj_name2id(model, mjOBJ_XBODY, "thorax");
  if (residual_.thorax_body_id_ < 0) mju_error("body 'thorax' not found");

  residual_.head_site_id_ = mj_name2id(model, mjOBJ_SITE, "head");
  if (residual_.head_site_id_ < 0) mju_error("site 'head' not found");

  // foot geom ids
  int foot_index = 0;
  for (const char* footname : {"tracking[claw_T1_left]", "tracking[claw_T1_right]",
                               "tracking[claw_T2_left]", "tracking[claw_T2_right]",
                               "tracking[claw_T3_left]", "tracking[claw_T3_right]"}) {
    int foot_id = mj_name2id(model, mjOBJ_SITE, footname);
    if (foot_id < 0) mju_error_s("geom '%s' not found", footname);
    residual_.foot_geom_id_[foot_index] = foot_id;
    foot_index++;
  }

  // foot geom ids
  int joint_index = 0;
  for (const auto &joint_name: joint_names) {
    int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
    if (joint_id < 0) mju_error_s("geom '%s' not found", joint_name.c_str());
    residual_.joint_geom_id_[joint_index] = joint_id;
    joint_index++;
  }

}

}  // namespace mjpc::fruitfly
