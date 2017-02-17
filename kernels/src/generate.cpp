#include <iostream>
#include <tuple>
#include <skepu2.hpp>

#include "../include/skepuimg.h"
	
struct ComplexFloat 
{
	float r, i;
};

namespace SkePUImageProcessing
{
	
	[[skepu::userconstant]] constexpr float
		CENTER_X = -.5f,
		CENTER_Y = 0.f;
	
	[[skepu::userconstant]] constexpr size_t
		MAX_ITERS = 255;
	
	
	ComplexFloat square_c(ComplexFloat c)
	{
		ComplexFloat r;
		r.r = c.r * c.r - c.i * c.i;
		r.i = c.i * c.r + c.r * c.i;
		return r;
	}
	
	float abs_sq_c(ComplexFloat c)
	{
		return c.r * c.r + c.i * c.i;
	}
	
	ComplexFloat add_c(ComplexFloat lhs, ComplexFloat rhs)
	{
		ComplexFloat r;
		r.r = lhs.r + rhs.r;
		r.i = lhs.i + rhs.i;
		return r;
	}
	
	auto mandelbroter = skepu2::Map<0>([](skepu2::Index2D index, skepu2::Vec<RGBPixel> colors, size_t height, size_t width, float scale, size_t maxiters) -> RGBPixel
	{
		RGBPixel r;
		ComplexFloat a;
		a.r = scale / height * (index.col - width/2.f) + CENTER_X;
		a.i = scale / height * (index.row - width/2.f) + CENTER_Y;
		ComplexFloat c = a;
		
		for (size_t i = 0; i < maxiters; ++i)
		{
			a = add_c(square_c(a), c);
			if (abs_sq_c(a) > 4)
				return colors.data[(size_t)((float)i / maxiters * colors.size)];
		}
		return colors.data[colors.size-1];
	});
	
	skepu2::Vector<RGBPixel> generateColors(size_t numColors)
	{
		// Gradient color
		RGBPixel start, stop;
		
		skepu2::Vector<RGBPixel> colors(numColors);
		
		// Start color
		start.red   = 219;
		start.green =  57;
		start.blue  =   0;
		
		// Stop color
		stop.red   = 0;
		stop.green = 0;
		stop.blue  = 0;
		
		// Initialize the color vector
		for (size_t i = 0; i < numColors; i++)
		{
			RGBPixel pixel;
			pixel.green = (stop.green - start.green) * ((double) i / numColors) + start.green;
			pixel.red   = (stop.red   - start.red  ) * ((double) i / numColors) + start.red;
			pixel.blue  = (stop.blue  - start.blue ) * ((double) i / numColors) + start.blue;
			colors[i] = pixel;
		}
		
		return colors;
	}
	
	float mandelbrot(skepu2::Matrix<RGBPixel> *img, float scale)
	{
		static skepu2::Vector<RGBPixel> colors = generateColors(64);
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			mandelbroter.setBackend(backendSpec());
			mandelbroter(*img, colors, img->total_rows(), img->total_cols(), scale, 50);
		});
		return time.count() / 1E6; // us -> s
	}
}
