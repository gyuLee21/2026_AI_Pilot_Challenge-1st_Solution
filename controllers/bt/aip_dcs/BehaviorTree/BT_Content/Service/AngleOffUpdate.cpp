//Target과의 AngleOff값을 업데이트 하는 서비스 노드

#include "AngleOffUpdate.h"

namespace Action
{
	PortsList AngleOffUpdate::providedPorts()
	{
		return {
			InputPort<CPPBlackBoard*>("BB")
		};
	}

	NodeStatus AngleOffUpdate::tick()
	{
		Optional<CPPBlackBoard*> BB = getInput<CPPBlackBoard*>("BB");

		Vector3 MFV = (*BB)->MyForwardVector;			//블랙보드에서 ForwardVector 불러옴
		Vector3 TFV = (*BB)->TargetForwardVector;		//블랙보드에서 적기의 ForwardVector 불러옴
		float dot = (*BB)->MyForwardVector.angleBetween((*BB)->TargetForwardVector);		//

		(*BB)->MyAngleOff_Degree = dot * 57.2958;


		return NodeStatus::SUCCESS;
	}

}