//AspectAngle 업데이트 하는 서비스 노드

#include "AspectAngleUpdate.h"

namespace Action
{
	PortsList AspectAngleUpdate::providedPorts()
	{
		return {
			InputPort<CPPBlackBoard*>("BB")
		};
	}

	NodeStatus AspectAngleUpdate::tick()
	{
		Optional<CPPBlackBoard*> BB = getInput<CPPBlackBoard*>("BB");

		Vector3 MyLocation = (*BB)->MyLocation_Cartesian;				//내 위치
		Vector3 TargetLocation = (*BB)->TargetLocaion_Cartesian;		//타겟 위치
		Vector3 TFV = (*BB)->TargetForwardVector;						//타겟의 Forward Vector
		Vector3 TUV = (*BB)->TargetUpVector;							//타겟의 Up Vector

		Vector3 TargetToMyPlane = MyLocation - TargetLocation;			//타겟위치에서 내위치까지의 벡터
		float P = TargetToMyPlane.dot(TUV);								//Up Vector방향 길이 구하기
		Vector3 Proj_MyLocation = MyLocation - P * TUV;					//내 위치를 적기의 Up 벡터를 법선 벡터를 가지고 적기의 위치를 지나는 평면으로 프로젝션

		Vector3 TPM = Proj_MyLocation - TargetLocation;					//프로젝션된 내 위치와 타겟사이의 벡터

		float AA = TPM.angleBetween(TFV);								//두 벡터 사이의 각 구하기 -> AA

		(*BB)->MyAspectAngle_Degree = AA* 57.2958;						//디그리 값으로 사용하기 위하여 변환

		return NodeStatus::SUCCESS;
	}

}