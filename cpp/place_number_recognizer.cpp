#include "place_number_recognizer.h"

#include "help_opencv.h"
#include "help_alg.h"

#include <algorithm>

#include <tesseract/baseapi.h>

std::pair<std::string, cv::Rect> recognize_place_number(const cv::Mat &image,
                                                        const Color &color,
                                                        const cv::Rect &number_plate_rect)
{
  double ratio = 0.6;
  cv::Mat small_image;
  cv::resize(image, small_image, cv::Size(), ratio, ratio);

  cv::Mat gray;
  cv::cvtColor(small_image, gray, CV_RGB2GRAY);

  cv::Mat blur;
  cv::medianBlur(gray, blur, 3);

  cv::Mat canny;
  cv::Canny(blur, canny, 80, 50);

  cv::Mat elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::Mat dilate_img;
  cv::dilate(canny, dilate_img, elem);

  // leave areas only like a square
  auto filled_areas = find_filled_areas(dilate_img, 255);
  filled_areas.erase(std::remove_if(filled_areas.begin(), filled_areas.end(),
                                    [&small_image](const std::pair<std::vector<cv::Point>, cv::Rect> &area)
  {
    double ratio = (double)area.second.width / area.second.height;
    return ratio < 0.7 || ratio > 1.3 ||
           area.second.x == 0 || area.second.y == 0 ||
           area.second.x + area.second.width == small_image.cols ||
           area.second.y + area.second.height == small_image.rows;
  }), filled_areas.end());

  // area must containt an area like itself
  filled_areas.erase(std::remove_if(filled_areas.begin(), filled_areas.end(),
                                    [&filled_areas](const std::pair<std::vector<cv::Point>, cv::Rect> &area1)
  {
    for (auto &area2: filled_areas)
      if (area2.second.contains(area1.second.tl()) &&
          area2.second.contains(area1.second.br()))
      {
        double ratio1 = (double)area1.second.width / area1.second.height;
        double ratio2 = (double)area2.second.width / area2.second.height;
        double ratio = ratio1 / ratio2;

        if (ratio > 0.9 && ratio < 1.1 &&
            (double)area2.second.height / area1.second.height < 1.4)
          return false;
      }

    return true;
  }), filled_areas.end());

  // area must not cointains an area
  filled_areas.erase(std::remove_if(filled_areas.begin(), filled_areas.end(),
                                    [&filled_areas](const std::pair<std::vector<cv::Point>, cv::Rect> &area1)
  {
    for (auto &area2: filled_areas)
      if (area1.second.contains(area2.second.tl()) &&
          area1.second.contains(area2.second.br()))
        return true;

    return false;
  }), filled_areas.end());

  // select nearest area to the number plate rect, it must be left of number plate
  cv::Rect scaled_number_plate_rect = number_plate_rect * ratio;
  std::pair<std::vector<cv::Point>, cv::Rect> nearest_area;
  int least_distance = -1;
  for (auto &area: filled_areas)
  {
    if (area.second.x > scaled_number_plate_rect.br().x)
      continue;

    int distance = abs(area.second.x + area.second.width / 2 - scaled_number_plate_rect.x - scaled_number_plate_rect.width / 2);
    if (least_distance > distance || least_distance == -1)
    {
      least_distance = distance;
      nearest_area = area;
    }
  }

  if (least_distance == -1)
    return std::make_pair("", cv::Rect());

  int half_width = nearest_area.second.width / 2;
  int half_height = nearest_area.second.height / 2;
  int center_x = nearest_area.second.x + half_width;
  int center_y = nearest_area.second.y + half_height;
  double k = 0.8;
  nearest_area.second.x = center_x - half_width * k;
  nearest_area.second.y = center_y - half_height * k;
  nearest_area.second.width *= k;
  nearest_area.second.height *= k;
  cv::Mat place_number_area = image(nearest_area.second * (1.0 / ratio));
  cv::Mat gray_place_number_area;
  cv::cvtColor(place_number_area, gray_place_number_area, CV_RGB2GRAY);
  cv::Mat threshold_place_number_area;
  cv::threshold(gray_place_number_area, threshold_place_number_area, 100, 255, CV_THRESH_BINARY_INV);
  cv::imshow("threshold_place_number_area", threshold_place_number_area);

  tesseract::TessBaseAPI tess_api;
  tess_api.Init("tessdata", "eng");
  tess_api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
  tess_api.SetVariable("tessedit_char_whitelist", "1234567890");

  tess_api.SetImage(threshold_place_number_area.ptr(),
                    threshold_place_number_area.size().width,
                    threshold_place_number_area.size().height,
                    threshold_place_number_area.elemSize(),
                    threshold_place_number_area.step1());
  char *text = tess_api.GetUTF8Text();
  std::string place_number = text;
  delete[] text;

  place_number.erase(std::remove_if(place_number.begin(),
                                    place_number.end(),
                                    [](int char_value)
  {
    return !isdigit(char_value);
  }), place_number.end());

  return std::make_pair(place_number, nearest_area.second);
}
