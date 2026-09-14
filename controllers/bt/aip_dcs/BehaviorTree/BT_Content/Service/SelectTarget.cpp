#include "SelectTarget.h"

namespace Action
{
	PortsList SelectTarget::providedPorts()
	{
		return {
			InputPort<CPPBlackBoard*>("BB")
		};
	}



	NodeStatus SelectTarget::tick()
	{
		Optional<CPPBlackBoard*> BB = getInput<CPPBlackBoard*>("BB");

		//std::cout << "Size : " << (*BB)->Enemy.size() << std::endl;

		//학생들은 1대1만 쓸꺼라 그냥 깡으로 타겟 지정
		if((*BB)->Enemy.size() > 0)
		{
			(*BB)->ACM = EF;

			(*BB)->TargetLocaion_Cartesian = (*BB)->Enemy.at(0).Location;
			(*BB)->TargetRotation_EDegree = (*BB)->Enemy.at(0).Rotation;
			(*BB)->TargetSpeed_MS = (*BB)->Enemy.at(0).Speed;

		}
		else
		{
			std::cout << "Target is not Valid!!!!!!" << std::endl;
		}

		return NodeStatus::SUCCESS;
	}

}