#include <iostream>
#include <tuple>
#include <algorithm>
#include <skepu2.hpp>

#include "../include/skepuimg.h"

namespace SkePUImageProcessing
{
	template<typename T>
	T max(T a, T b)
	{
		return (a > b) ? a : b;
	}
	
	template<typename T>
	T min(T a, T b)
	{
		return (a < b) ? a : b;
	}
	
	unsigned char intensity(RGBPixel input)
	{
		return ((unsigned int)input.red + (unsigned int)input.green + (unsigned int)input.blue) / 3;
	}
	
	auto rgb_to_grayscale = skepu2::Map([](RGBPixel input) -> GrayscalePixel
	{
		GrayscalePixel output;
		output.intensity = intensity(input);
		return output;
	});
	
	auto grayscale_to_rgb = skepu2::Map([](GrayscalePixel input) -> RGBPixel
	{
		RGBPixel output;
		output.red   = input.intensity;
		output.green = input.intensity;
		output.blue  = input.intensity;
		return output;
	});
	
	auto desaturate_kernel_old = skepu2::Map([](RGBPixel input) -> RGBPixel
	{
		RGBPixel output;
		unsigned char in = intensity(input);
		output.red   = in;
		output.green = in;
		output.blue  = in;
		return output;
	});
	
	auto desaturate_kernel = skepu2::Map([](RGBPixel input, float saturation) -> RGBPixel
	{
		RGBPixel output;
		unsigned char in = intensity(input) * (1 - saturation);
		output.red   = input.red   * saturation + in;
		output.green = input.green * saturation + in;
		output.blue  = input.blue  * saturation + in;
		return output;
	});
	
	auto bw_kernel = skepu2::Map([](RGBPixel input) -> RGBPixel
	{
		RGBPixel output;
		unsigned char in = (intensity(input) > 127) ? 255 : 0;
		output.red   = in;
		output.green = in;
		output.blue  = in;
		return output;
	});
	
	float desaturate(skepu2::Matrix<RGBPixel> *img, float saturation)
	{
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			desaturate_kernel.setBackend(backendSpec());
			desaturate_kernel(*img, *img, saturation);
		});
		return time.count() / 1E6; // us -> s
	}
	
	float blackwhite(skepu2::Matrix<RGBPixel> *img)
	{
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			bw_kernel.setBackend(backendSpec());
			bw_kernel(*img, *img);
		});
		return time.count() / 1E6; // us -> s
	}


	
	auto convolution_grayscale = skepu2::MapOverlap([](int o, size_t stride, const GrayscalePixel *image, const skepu2::Vec<float> filter, float offset, float scaling) -> GrayscalePixel
	{
		GrayscalePixel result;
		float intensity = 0;
		
		for (int i = -o; i <= o; i++)
		{
			GrayscalePixel p = image[i*stride];
			intensity += p.intensity * filter[i+o];
		}
		
		result.intensity = (intensity + offset) * scaling;
		return result;
	});
	
	auto convolution_rgb = skepu2::MapOverlap([](int o, size_t stride, const RGBPixel *image, const skepu2::Vec<float> filter, float offset, float scaling) -> RGBPixel
	{
		float r = 0, g = 0, b = 0;
		for (int i = -o; i <= o; i++)
		{
			RGBPixel p = image[i*stride];
			r += p.red   * filter[i+o];
			g += p.green * filter[i+o];
			b += p.blue  * filter[i+o];
		}
		
		RGBPixel res;
		res.red   = min(max(0.f, (r + offset) * scaling), 255.f);
		res.green = min(max(0.f, (g + offset) * scaling), 255.f);
		res.blue  = min(max(0.f, (b + offset) * scaling), 255.f);
		return res;
	});
	
	auto filter_gen = skepu2::Map<0>([](skepu2::Index1D index, size_t r, float sigma) -> float
	{
		const float pi = 3.141592;
		float i = (float)index.i - r;
		return exp(-i*i / (2 * sigma * sigma)) / sqrt(2* pi * sigma * sigma);
	});
	
	auto distance = skepu2::Map<2>([](GrayscalePixel x, GrayscalePixel y) -> GrayscalePixel
	{
		GrayscalePixel result;
		float xf = (float)x.intensity / 255 - 0.5;
		float yf = (float)y.intensity / 255 - 0.5;
		result.intensity = 255 - sqrt(xf*xf + yf*yf) * 2 * 255;
		return result;
	});
	
	auto convkernels = std::tie(convolution_grayscale, convolution_rgb);
	
	template<typename PixelType>
	float gaussian(skepu2::Matrix<PixelType> *img, float blur_sigma)
	{
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			auto convolution = std::get<PixelTypeID<PixelType>::value>(convkernels);
			convolution.setBackend(backendSpec());
			filter_gen.setBackend(backendSpec());
			
			const size_t blur_radius = ceil(3.0 * blur_sigma);
			skepu2::Vector<float> filter(blur_radius * 2 + 1);
			filter_gen(filter, blur_radius, blur_sigma);
			float sum = 0;
			for (float f : filter) sum += f;
			
			convolution.setOverlap(blur_radius);
			convolution.setEdgeMode(skepu2::Edge::Duplicate);
			convolution.setOverlapMode(skepu2::Overlap::RowColWise);
			convolution(*img, *img, filter, 0, 1.0);
		});
		return time.count() / 1E6; // us -> s
	}
	
	
	float edge_gray(skepu2::Matrix<GrayscalePixel> *img)
	{
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			convolution_grayscale.setBackend(backendSpec());
			distance.setBackend(backendSpec());
			
			skepu2::Matrix<GrayscalePixel>
				tempA(img->total_rows(), img->total_cols()),
				tempB(img->total_rows(), img->total_cols());
				
			// Sobel edge detection
			skepu2::Vector<float> avrg_filter {  1.0, 2.0, 1.0 };
			skepu2::Vector<float> diff_filter { -1.0, 0.0, 1.0 };
			convolution_grayscale.setOverlap(1);
			
			convolution_grayscale.setOverlapMode(skepu2::Overlap::RowWise);
			convolution_grayscale(tempA, *img, diff_filter, 255.0, 0.50); // x-dir
			convolution_grayscale(tempB, *img, avrg_filter,   0.0, 0.25); // y-dir
			
			convolution_grayscale.setOverlapMode(skepu2::Overlap::ColWise);
			convolution_grayscale(*img,  tempA, avrg_filter,   0.0, 0.25); // x-dir
			convolution_grayscale(tempA, tempB, diff_filter, 255.0, 0.50); // y-dir
			
			// Final computation
			distance(*img, *img, tempA);
		});
		return time.count() / 1E6; // us -> s
	}
	
	float edge_rgb(skepu2::Matrix<RGBPixel> *img)
	{
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			rgb_to_grayscale.setBackend(backendSpec());
			grayscale_to_rgb.setBackend(backendSpec());
			
			skepu2::Matrix<GrayscalePixel> temp(img->total_rows(), img->total_cols());
			
			rgb_to_grayscale(temp, *img);
			edge_gray(&temp);
			grayscale_to_rgb(*img, temp);
		});
		return time.count() / 1E6; // us -> s
	}
	
	
	skepu2::BackendSpec spec;
	
	skepu2::BackendSpec backendSpec()
	{
		return spec;
	}
	
	void setBackend(std::string backend)
	{
		spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(backend)};
	}
	
	std::string getBackend()
	{
		std::stringstream ss;
		ss << spec.backend();
		return ss.str();
	}
	
	void setCPUThreads(size_t numThreads)
	{
		spec.setCPUThreads(std::min<size_t>(std::max<size_t>(1, numThreads), 32));
	}
	
	size_t getCPUThreads()
	{
		return spec.CPUThreads();
	}
	
	
	template float gaussian(skepu2::Matrix<GrayscalePixel> *img, float blur_sigma);
	template float gaussian(skepu2::Matrix<RGBPixel> *img, float blur_sigma);
}
