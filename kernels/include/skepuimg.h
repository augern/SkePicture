#pragma once

#include <iostream>
//#include <skepu2.hpp>

struct GrayscalePixel
	{
		unsigned char intensity;
	};
	
	struct RGBPixel
	{
		unsigned char red, green, blue;
	};
	
namespace SkePUImageProcessing
{
	
	
	inline std::ostream &operator<<(std::ostream &o, RGBPixel pixel)
	{
		o << "{" << (int)pixel.red << "|" << (int)pixel.green << "|" << (int)pixel.blue << "}";
		return o;
	}
	
	
	
	template<typename T> struct PixelTypeID;
	template<> struct PixelTypeID<GrayscalePixel>: std::integral_constant<size_t, 0> {};
	template<> struct PixelTypeID<RGBPixel>: std::integral_constant<size_t, 1> {};
	
	template<typename PixelType>
	float gaussian(skepu2::Matrix<PixelType> *img, float blur_sigma);
	
	float median(skepu2::Matrix<RGBPixel> *img, size_t radius);
	
	float edge_gray(skepu2::Matrix<GrayscalePixel> *img);
	float edge_rgb(skepu2::Matrix<RGBPixel> *img);
	
	float stencil(skepu2::Matrix<RGBPixel> *img, skepu2::Matrix<float> *stencil, float scaling);
	

	float desaturate(skepu2::Matrix<RGBPixel> *img, float saturation);
	float blackwhite(skepu2::Matrix<RGBPixel> *img);
	
	template<typename PixelType>
	float invert(skepu2::Matrix<PixelType> *img);
	
	float hue(skepu2::Matrix<RGBPixel> *img, float hue);
	
	
	// Geneators
//	float mandelbrot(skepu2::Matrix<GrayscalePixel> *img, float scale);
	float mandelbrot(skepu2::Matrix<RGBPixel> *img, float scale);
	
	
	
	void setBackend(std::string backend);
	std::string getBackend();
	
	skepu2::BackendSpec backendSpec();
	void setCPUThreads(size_t numThreads);
	size_t getCPUThreads();
	
}