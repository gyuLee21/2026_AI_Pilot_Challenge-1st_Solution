//아군기와 적기의 자세를 이용하여 각각의 Forward, Up, Right 방향의 벡터를 구하는 서비스 노드

#include "DirectionVectorUpdate.h"

namespace Action
{
	PortsList DirectionVectorUpdate::providedPorts()
	{
		return {
			InputPort<CPPBlackBoard*>("BB")
		};
	}

	NodeStatus DirectionVectorUpdate::tick()
	{
		Optional<CPPBlackBoard*> BB = getInput<CPPBlackBoard*>("BB");

		EulerAngle MyRotation = (*BB)->MyRotation_EDegree;

		//아군기
		MyRotation = MyRotation / 57.2958;
		Quaternion MyQU = MyRotation.toQuaternion();

		//쿼터니언을 이용하여정면벡터(ForwardVector)를 생성
		Vector3 MyForwardVector;
		MyForwardVector.Y = 2 * (MyQU.X*MyQU.Z + MyQU.W * MyQU.Y);
		MyForwardVector.Z = -2 * (MyQU.Y*MyQU.Z - MyQU.W *  MyQU.X);
		MyForwardVector.X = 1 - 2 * (MyQU.X*MyQU.X + MyQU.Y * MyQU.Y);

		//쿼터니언을 이용하여 수직벡터(UpVector)를 생성
		Vector3 MyUpVector;
		MyUpVector.X = -2 * (MyQU.Y*MyQU.Z + MyQU.W * MyQU.X);
		MyUpVector.Y = -2 * (MyQU.X*MyQU.Y - MyQU.W * MyQU.Z);
		MyUpVector.Z = 1 - 2 * (MyQU.X*MyQU.X + MyQU.Z * MyQU.Z);

		//쿼터니언을 이용하여 오른쪽벡터(RightVector)를 생성
		Vector3 MyRightVector;
		MyRightVector.X = 2 * (MyQU.X*MyQU.Z - MyQU.W * MyQU.Y);
		MyRightVector.Y = 1 - 2 * (MyQU.Y*MyQU.Y + MyQU.Z * MyQU.Z);
		MyRightVector.Z = -2 * (MyQU.X*MyQU.Y + MyQU.W * MyQU.Z);

		(*BB)->MyForwardVector = MyForwardVector;
		(*BB)->MyUpVector = MyUpVector;
		(*BB)->MyRightVector = MyRightVector;

		//적기
		EulerAngle TargetRotation = (*BB)->TargetRotation_EDegree;

		TargetRotation = TargetRotation / 57.2958;
		Quaternion TaQU = TargetRotation.toQuaternion();
		//쿼터니언을 이용하여정면벡터(ForwardVector)를 생성
		Vector3 TargetForwardVector;
		TargetForwardVector.Y = 2 * (TaQU.X*TaQU.Z + TaQU.W * TaQU.Y);
		TargetForwardVector.Z = -2 * (TaQU.Y*TaQU.Z - TaQU.W *  TaQU.X);
		TargetForwardVector.X = 1 - 2 * (TaQU.X*TaQU.X + TaQU.Y * TaQU.Y);

		//쿼터니언을 이용하여 수직벡터(UpVector)를 생성
		Vector3 TargetUpVector;
		TargetUpVector.X = -2 * (TaQU.Y*TaQU.Z + TaQU.W * TaQU.X);
		TargetUpVector.Y = -2 * (TaQU.X*TaQU.Y - TaQU.W * TaQU.Z);
		TargetUpVector.Z = 1 - 2 * (TaQU.X*TaQU.X + TaQU.Z * TaQU.Z);

		//쿼터니언을 이용하여 오른쪽벡터(RightVector)를 생성
		Vector3 TargetRightVector;
		TargetRightVector.X = 2 * (TaQU.X*TaQU.Z - TaQU.W * TaQU.Y);
		TargetRightVector.Y = 1 - 2 * (TaQU.Y*TaQU.Y + TaQU.Z * TaQU.Z);
		TargetRightVector.Z = -2 * (TaQU.X*TaQU.Y + TaQU.W * TaQU.Z);

		(*BB)->TargetForwardVector = TargetForwardVector;
		(*BB)->TargetUpVector = TargetUpVector;
		(*BB)->TargetRightVector = TargetRightVector;


		//BTFunc::AddNodeExcute(&(*BB)->Temp_Text, std::string("Direction Vector Update"));
		return NodeStatus::SUCCESS;
	}

}