#ifndef ANPR_H
#define ANPR_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

namespace ANPR
{
  std::string find_and_recognize_number_plate(const cv::Mat &image,
                                              const cv::Rect &search_rect);
}


#endif // ANPR_H
