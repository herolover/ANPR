#ifndef ANPR_H
#define ANPR_H

#include <string>

#include <opencv2/opencv.hpp>

namespace ANPR
{
  std::string recognize_number_plate(const cv::Mat &number_plate_image);
}


#endif // ANPR_H
