#include "Functions.h"

namespace BTFunc
{
	void AddNodeExcute(std::string * out, std::string input)
	{
		out->append(input);
		out->append("\n");
	}
	void SaveTextData(std::string * tempString, std::string * BT_Text)
	{
		if (tempString != nullptr && BT_Text != nullptr)
		{
			if (tempString->length() > 910)
				tempString->clear();

			BT_Text->clear();

			BT_Text->append((*tempString));
			tempString->clear();
		}
	}

}