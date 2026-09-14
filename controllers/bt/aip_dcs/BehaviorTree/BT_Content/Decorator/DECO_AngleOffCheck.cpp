#include "DECO_AngleOffCheck.h"

namespace Action
{
	PortsList DECO_AngleOffCheck::providedPorts()
	{
		return {
			InputPort<CPPBlackBoard*>("BB"),
			InputPort<std::string>("UpDown"),
			InputPort<std::string>("InputAO")
		};
	}

	NodeStatus DECO_AngleOffCheck::tick()
	{
		Optional<CPPBlackBoard*> BB = getInput<CPPBlackBoard*>("BB");
		Optional<std::string> UpOrDown = getInput<std::string>("UpDown");
		Optional<std::string> IAO = getInput<std::string>("InputAO");

		float CurrentAO = (*BB)->MyAngleOff_Degree;
		std::string UD = UpOrDown.value();
		float InputAngleOff = std::stof(IAO.value());

		if (UD == "Greater")
		{
			if (CurrentAO >= InputAngleOff)
			{

				return NodeStatus::SUCCESS;
			}
			else
				return NodeStatus::FAILURE;
		}
		else if (UD == "Less")
		{
			if (CurrentAO <= InputAngleOff)
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