#include "ANPR.h"

#include <string>

#include <boost/format.hpp>
#include <boost/regex.hpp>

#include <tesseract/baseapi.h>

#include "help_alg.h"
#include "help_opencv.h"


std::string ANPR::recognize_number_plate(const cv::Mat &number_plate_image)
{
  cv::Mat proc_image = convert_to_grayscale_and_remove_noise(number_plate_image);

//  cv::imshow("proc_image", proc_image);

  cv::Mat blured_image;
  cv::GaussianBlur(proc_image, blured_image, cv::Size(101, 101), 0);

  cv::Mat equalized_image;
  cv::divide(proc_image, blured_image, equalized_image, 256.0);

//  cv::imshow("equalized_image", equalized_image);

  cv::Mat threshold_image;
  cv::threshold(equalized_image, threshold_image, 210.0, 255.0, CV_THRESH_BINARY_INV);

//  cv::imshow("threshold_image_prev", threshold_image);

  std::vector<std::vector<cv::Point> > areas = find_filled_areas(threshold_image.clone(), 255);

  // remove too wide or too small areas
  areas.erase(std::remove_if(areas.begin(), areas.end(),
                             [threshold_image](const std::vector<cv::Point> &area)
  {
    cv::Rect area_bound = cv::boundingRect(area);

    const double k0 = 1.1;
    const double k1 = 0.25;

    double ratio = (double)area_bound.width / area_bound.height;

    return ratio > k0 || ratio < k1 ||
//           (double)area.size() / threshold_image.total() < 0.006 ||
           area_bound.height == threshold_image.size().height ||
           area_bound.x == 0 || area_bound.y == 0 ||
           area_bound.x + area_bound.width == threshold_image.size().width ||
           area_bound.y + area_bound.height == threshold_image.size().height;
  }), areas.end());

  int max_height = 0;
  int min_height = std::numeric_limits<int>::max();
  for (auto &area: areas) {
    cv::Rect area_bound = cv::boundingRect(area);
    max_height = std::max(max_height, area_bound.height);
    min_height = std::min(min_height, area_bound.height);
  }

  double height_threshold_coef = 0.5;
  int height_threshold = (int)((max_height - min_height) * height_threshold_coef) + min_height;
  areas.erase(std::remove_if(areas.begin(), areas.end(),
                             [height_threshold](const std::vector<cv::Point> &area)
  {
    cv::Rect area_bound = cv::boundingRect(area);
    return area_bound.height < height_threshold;
  }), areas.end());

  threshold_image = cv::Mat(threshold_image.size(), threshold_image.type());
  for (auto &area: areas)
    draw_area(threshold_image, area, 255);

//  cv::imshow("threshold_image", threshold_image);

  tesseract::TessBaseAPI tess_api;
  tess_api.Init("tessdata", "eng");
  tess_api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
  tess_api.SetVariable("tessedit_char_whitelist", "ABCEHKMOPTXY1234567890");

  tess_api.SetImage(threshold_image.ptr(),
                    threshold_image.size().width,
                    threshold_image.size().height,
                    threshold_image.elemSize(),
                    threshold_image.step1());
  char *text = tess_api.GetUTF8Text();
  std::string number_plate_text = text;
  delete[] text;

  number_plate_text.erase(std::remove_if(number_plate_text.begin(),
                                         number_plate_text.end(),
                                         isspace),
                          number_plate_text.end());


  boost::match_results<std::string::iterator> what;
  if (boost::regex_search(number_plate_text.begin(),
                          number_plate_text.end(),
                          what,
                          boost::regex("[ABCEHKMOPTXY0][[:digit:]]{3}[ABCEHKMOPTXY0]{2}[[:digit:]]{2,3}")))
  {
    int zero_test_indexes[] = {0, 4, 5};
    for (auto index: zero_test_indexes)
      if (*(what[0].first + index) == '0')
        *(what[0].first + index) = 'O';

    number_plate_text = std::string(what[0].first, what[0].second);
  }

  return number_plate_text;
}
