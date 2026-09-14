#include "DECO_DistanceCheck.h"

namespace Action
{
	PortsList DECO_DistanceCheck::providedPorts()
	{
		return {
			InputPort<CPPBlackBoard*>("BB"),
			InputPort<std::string>("UpDown"),
			InputPort<std::string>("Distance")
		};
	}

	NodeStatus DECO_DistanceCheck::tick()
	{
		Optional<CPPBlackBoard*> BB = getInput<CPPBlackBoard*>("BB");
		Optional<std::string> UpOrDown = getInput<std::string>("UpDown");
		Optional<std::string> Dist = getInput<std::string>("Distance");

		float CurrentDistance = (*BB)->Distance;
		std::string UD = UpOrDown.value();
		float InputDistance = std::stof(Dist.value());



		if (UD == "Greater")
		{
			if (CurrentDistance >= InputDistance)
			{
				return NodeStatus::SUCCESS;
			}
			else
				return NodeStatus::FAILURE;
		}
		else if (UD == "Less")
		{
			if (CurrentDistance <= InputDistance)
			{

				return NodeStatus::SUCCESS;
			}
			else
				return NodeStatus::FAILURE;
		}
		else
		{
			//UpDown 입력 문자열이 오타난건 아닌지 확인 필요!!!! Greater 나 Less 가 아님
			return NodeStatus::FAILURE;
		}
	}
}