#include "utils/utils.hpp"
using namespace minicombust::utils;

template<class T>
class gpu_Face
{
	private:

	public:
		T cell0;
		T cell1;

		gpu_Face(T cell0, T cell1) : cell0(cell0), cell1(cell1)
		{ }
}; // class face
