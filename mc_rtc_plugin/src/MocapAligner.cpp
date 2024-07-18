#include "MocapAligner.h"

#include <mc_control/GlobalPluginMacros.h>
#include <mc_state_observation/conversions/kinematics.h>

namespace mc_plugin
{

MocapAligner::~MocapAligner() = default;

void MocapAligner::init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config)
{
  mc_rtc::log::info("MocapAligner::init called with configuration:\n{}", config.dump(true, true));

  robot_ = config("robot", controller.controller().robot().name());
  config("bodyName", mocapBodyName_);

  auto & logger = (const_cast<mc_control::MCController &>(controller.controller())).logger();
  mc_state_observation::conversions::kinematics::addToLogger(logger, worldMocapBodyKine_, "MocapAligner_worldBodyKine");
}

void MocapAligner::reset(mc_control::MCGlobalController & controller)
{
  mc_rtc::log::info("MocapAligner::reset called");
}

void MocapAligner::before(mc_control::MCGlobalController &)
{
  mc_rtc::log::info("MocapAligner::before");
}

void MocapAligner::after(mc_control::MCGlobalController & controller)
{
  mc_rtc::log::info("MocapAligner::after");

  worldMocapBodyKine_ = mc_state_observation::conversions::kinematics::fromSva(
      controller.controller().realRobot(robot_).bodyPosW(mocapBodyName_),
      stateObservation::kine::Kinematics::Flags::pose);
}

mc_control::GlobalPlugin::GlobalPluginConfiguration MocapAligner::configuration()
{
  mc_control::GlobalPlugin::GlobalPluginConfiguration out;
  out.should_run_before = true;
  out.should_run_after = true;
  out.should_always_run = true;
  return out;
}

} // namespace mc_plugin

EXPORT_MC_RTC_PLUGIN("MocapAligner", mc_plugin::MocapAligner)
