#ifndef ANPR_H
#define ANPR_H

#include <string>
#include <utility>

#include <opencv2/opencv.hpp>

namespace ANPR
{
  std::pair<std::string, cv::Rect> recognize_number_plate(const cv::Mat &image,
                                                          const cv::Rect &search_rect);
}


#endif // ANPR_H
