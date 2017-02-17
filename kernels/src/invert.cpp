#include <iostream>
#include <tuple>
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
	
	auto invert_rgb = skepu2::Map<1>([](RGBPixel input)
	{
		RGBPixel output;
		output.red   = 255 - input.red;
		output.green = 255 - input.green;
		output.blue  = 255 - input.blue;
		return output;
	});
	
	auto invert_grayscale = skepu2::Map<1>([](GrayscalePixel input)
	{
		GrayscalePixel output;
		output.intensity = 255 - input.intensity;
		return output;
	});
	
	float clampf(float min, float val, float max)
	{
		return (val > max) ? max : ((val < min) ? min : val);
	}
	
	
	
	auto hue_rgb = skepu2::Map<1>([](RGBPixel input, float H)
	{
		// input RGB values
		float R = input.red   / 255.0f;
		float G = input.green / 255.0f;
		float B = input.blue  / 255.0f;
		
		// Transform to HSI color space
		float I     = (R + G + B) / 3.0f;
		float alpha = 1.5f * (R - I);
		float beta  = sqrt(3.0f) * 0.5f * (G - B);
	//	float H2 = atan2(beta, alpha);
		float C     = sqrt(alpha * alpha + beta * beta);
		
		// Transform back to RGB
		beta  = sin(H) * C;
		alpha = cos(H) * C;
		
		// update RGB values
		RGBPixel output;
		output.red   = clampf(0.f, I + (2.0f / 3.0f) * alpha, 1.f) * 255.0;
		output.green = clampf(0.f, I - alpha / 3.0f + beta / sqrt(3.0f), 1.f) * 255.0;
		output.blue  = clampf(0.f, I - alpha / 3.0f - beta / sqrt(3.0f), 1.f) * 255.0;
		return output;
	});
	
	
	template<typename PixelType>
	float invert(skepu2::Matrix<PixelType> *img)
	{
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			auto kernel = std::get<PixelTypeID<PixelType>::value>(std::tie(invert_grayscale, invert_rgb));
			kernel.setBackend(backendSpec());
			kernel(*img, *img);
		});
		return time.count() / 1E6; // us -> s
	}
	
	float hue(skepu2::Matrix<RGBPixel> *img, float hue)
	{
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			hue_rgb.setBackend(backendSpec());
			hue_rgb(*img, *img, hue);
		});
		return time.count() / 1E6; // us -> s
	}
	
	unsigned char median_helper(int ox, int oy, size_t stride, const RGBPixel *image [[skepu::overlapArrayForward]], size_t offset)
	{
		long fineHistogram[256], coarseHistogram[16];
		
		for (int i = 0; i < 256; i++)
			fineHistogram[i] = 0;
		
		for (int i = 0; i < 16; i++)
			coarseHistogram[i] = 0;
		
		for (int row = -oy; row <= oy; row++)
		{
			for (int column = -ox; column <= ox; column += 1)
			{ 
				unsigned char imageValue = ((unsigned char *)(&image[row * stride + column]))[offset];
				fineHistogram[imageValue]++;
				coarseHistogram[imageValue / 16]++;
			}
		}
		
		int count = 2 * oy * (oy + 1);
		
		unsigned char coarseIndex;
		for (coarseIndex = 0; coarseIndex < 16; ++coarseIndex)
		{
			if ((long)count - coarseHistogram[coarseIndex] < 0) break;
			count -= coarseHistogram[coarseIndex];
		}
		
		unsigned char fineIndex = coarseIndex * 16;
		while ((long)count - fineHistogram[fineIndex] >= 0)
			count -= fineHistogram[fineIndex++];
		
		return fineIndex;
	}
	
	RGBPixel median_kernel(int ox, int oy, size_t stride, const RGBPixel *image)
	{
		RGBPixel retval;
		retval.red   = median_helper(ox, oy, stride, image, 0);
		retval.green = median_helper(ox, oy, stride, image, 1);
		retval.blue  = median_helper(ox, oy, stride, image, 2);
		return retval;
	}
	auto calculateMedian = skepu2::MapOverlap(median_kernel);
	
	template<typename T>
	T oversample_kernel(skepu2::Index2D index, const skepu2::Mat<T> img, size_t radiusY, size_t radiusX)
	{
		size_t x = min(max(radiusX, index.col), img.cols + radiusX - 1) - radiusX;
		size_t y = min(max(radiusY, index.row), img.rows + radiusY - 1) - radiusY;
		return img.data[y * img.cols + x];
	}
	
	auto oversampler = skepu2::Map<0>(oversample_kernel<RGBPixel>);
	
	template<typename T>
	skepu2::Matrix<T> oversample(skepu2::Matrix<T> &img, size_t radiusY, size_t radiusX)
	{
		// Create a matrix which fits the image and the padding needed.
		skepu2::Matrix<T> res(img.total_rows() + 2 * radiusY, img.total_cols() + 2 * radiusX);
		oversampler.setBackend(backendSpec());
		oversampler(res, img, radiusY, radiusX);
		return res;
	}
	
	float median(skepu2::Matrix<RGBPixel> *img, size_t radius)
	{
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			skepu2::Matrix<RGBPixel> temp = oversample(*img, radius, radius);
			std::cout << temp.total_cols() << "\n";
			calculateMedian.setBackend(backendSpec());
			calculateMedian.setOverlap(radius);
			calculateMedian(*img, temp);
		});
		return time.count() / 1E6; // us -> s
	}
	
	
	auto stencil_kernel = skepu2::MapOverlap([](int ox, int oy, size_t stride, const RGBPixel *m, const skepu2::Mat<float> filter, float scaling) -> RGBPixel
	{
		float red = 0, green = 0, blue = 0;
		for (int y = -oy; y <= oy; ++y)
			for (int x = -ox; x <= ox; ++x)
			{
				RGBPixel elem = m[y*(int)stride+x];
				float coeff = filter.data[(y+oy)*(2*ox+1) + (x+ox)];
				red   += elem.red   * coeff;
				green += elem.green * coeff;
				blue  += elem.blue  * coeff;
			}
		
		RGBPixel res;
		res.red   = min(max(0.f, red   * scaling), 255.f);
		res.green = min(max(0.f, green * scaling), 255.f);
		res.blue  = min(max(0.f, blue  * scaling), 255.f);
		return res;
	});
	
	float stencil(skepu2::Matrix<RGBPixel> *img, skepu2::Matrix<float> *stencil, float scaling)
	{
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			size_t overlapX = (stencil->total_cols() - 1) / 2;
			size_t overlapY = (stencil->total_rows() - 1) / 2;
			skepu2::Matrix<RGBPixel> temp = oversample(*img, overlapY, overlapX);
			
			stencil_kernel.setBackend(backendSpec());
			stencil_kernel.setOverlap(overlapX, overlapY);
			stencil_kernel(*img, temp, *stencil, scaling);
		});
		return time.count() / 1E6; // us -> s
	}
	
	
	template float invert(skepu2::Matrix<GrayscalePixel> *img);
	template float invert(skepu2::Matrix<RGBPixel> *img);
}
