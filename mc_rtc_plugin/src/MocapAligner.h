/*
 * Copyright 2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_control/GlobalPlugin.h>

#include <state-observation/tools/rigid-body-kinematics.hpp>

namespace mc_plugin
{

struct MocapAligner : public mc_control::GlobalPlugin
{
  void init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config) override;

  void reset(mc_control::MCGlobalController & controller) override;

  void before(mc_control::MCGlobalController &) override;

  void after(mc_control::MCGlobalController & controller) override;

  mc_control::GlobalPlugin::GlobalPluginConfiguration configuration() override;

  ~MocapAligner() override;

private:
  std::string robot_; // name of the robot

  std::string mocapBodyName_;
  stateObservation::kine::Kinematics worldMocapBodyKine_;
};

} // namespace mc_plugin
