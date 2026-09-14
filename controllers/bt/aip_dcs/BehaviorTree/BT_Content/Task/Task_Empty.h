#pragma once
/*
 그냥 기본 양식이라고 생각하고 얘를 복붙해가면서 쓰세요

*/
#include "../../behaviortree_cpp_v3\action_node.h"
#include "../../behaviortree_cpp_v3/bt_factory.h"
#include "../../../Geometry/Vector3.h"
#include "../Functions.h"
#include "../BlackBoard/CPPBlackBoard.h"

using namespace BT;

namespace Action
{
	class Task_Empty : public SyncActionNode
	{
	private:


	public:


		Task_Empty(const std::string& name, const NodeConfiguration& config) : SyncActionNode(name, config)
		{
		}

		~Task_Empty()
		{

		}

		static PortsList providedPorts();

		NodeStatus tick() override;
	};
}