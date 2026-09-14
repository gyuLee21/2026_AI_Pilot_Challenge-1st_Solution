#include "Task_Empty.h"

PortsList Action::Task_Empty::providedPorts()
{
	return {
			InputPort<CPPBlackBoard*>("BB")
	};
}

NodeStatus Action::Task_Empty::tick()
{
	Optional<CPPBlackBoard*> BB = getInput<CPPBlackBoard*>("BB");

	(*BB)->VP_Cartesian = (*BB)->MyLocation_Cartesian +  (*BB)->MyForwardVector * 10000;

	return NodeStatus::SUCCESS;
}
