#pragma once
#include "../../behaviortree_cpp_v3/action_node.h"
#include "../../behaviortree_cpp_v3/bt_factory.h"
#include "../../../Geometry/Vector3.h"
#include "../Functions.h"
#include "../BlackBoard/CPPBlackBoard.h"

using namespace BT;

namespace Action
{
	class DistanceUpdate : public SyncActionNode
	{
	private:


	public:


		DistanceUpdate(const std::string& name, const NodeConfiguration& config) : SyncActionNode(name, config)
		{
		}

		~DistanceUpdate()
		{

		}

		static PortsList providedPorts();

		NodeStatus tick() override;
	};
}
