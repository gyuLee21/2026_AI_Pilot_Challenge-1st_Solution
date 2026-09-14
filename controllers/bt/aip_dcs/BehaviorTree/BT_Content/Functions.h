#pragma once
#include "../../Geometry/Vector3.h"
#include "../../Geometry/EulerAngle.h"
#include "../../Geometry/Quaternion.h"
#include <vector>

using namespace BT_Geometry;

namespace BTFunc
{
	/*
	국과연에서 요구한 비헤비어트리 결정 과정을 보여주기 위해 각 노드에서 실행 과정(결과)를 문자열로 저장하기 위한 함수
	기존 문자열, 추가할 문자열
	*/
	void AddNodeExcute(std::string * out, std::string input);
	void SaveTextData(std::string * tempString, std::string * BT_Text);


}