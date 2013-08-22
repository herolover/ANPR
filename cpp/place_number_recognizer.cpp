#include "place_number_recognizer.h"

#include "help_opencv.h"

#include <algorithm>

#include <tesseract/baseapi.h>


std::string recognize_place_number(const cv::Mat &image, const Color &color)
{
  double ratio = 0.3;
  cv::Mat small_image;
  cv::resize(image, small_image, cv::Size(), ratio, ratio);

  cv::Mat blured_image;
  cv::medianBlur(small_image, blured_image, 5);

  cv::Mat difference_image(blured_image.size(), CV_8UC1);
  for (int i = 0; i < blured_image.cols; ++i)
    for (int j = 0; j < blured_image.rows; ++j)
    {
      cv::Vec<int, 3> diff = (cv::Vec<int, 3>)blured_image.at<Color>(j, i) - (cv::Vec<int, 3>)color;

      double value = 255.0 * cv::norm(diff) / sqrt(255.0 * 255.0 + 255.0 * 255.0 + 255.0 * 255.0);

      difference_image.at<unsigned char>(j, i) = (unsigned char)value;
    }

  cv::imshow("diff", difference_image);

  cv::Mat threshold_image;
  cv::threshold(difference_image, threshold_image, 60.0, 255.0, CV_THRESH_BINARY_INV);

  cv::imshow("threshold_image", threshold_image);

  auto filled_areas = find_filled_areas(threshold_image.clone(), 255);

  double image_center_x = small_image.cols * 0.5;
  filled_areas.erase(std::remove_if(filled_areas.begin(), filled_areas.end(),
                                    [&image_center_x](const std::vector<cv::Point> &area)
  {
    cv::Rect area_bound = cv::boundingRect(area);

    return area_bound.width / (double)area_bound.height > 2 ||
           area_bound.height / (double)area_bound.width > 2 ||
           area.size() < 40 || area_bound.x > image_center_x;
  }), filled_areas.end());

  if (filled_areas.size() == 0)
  {
    return "";
  }

  auto nearest_area_it = std::max_element(filled_areas.begin(), filled_areas.end(),
                                          [](const std::vector<cv::Point> &a,
                                             const std::vector<cv::Point> &b)
  {
    double a_center_x = 0.0;
    for (auto &point: a)
    {
      a_center_x += point.x;
    }
    a_center_x /= a.size();

    double b_center_x = 0.0;
    for (auto &point: b)
    {
      b_center_x += point.x;
    }
    b_center_x /= b.size();

    return a_center_x < b_center_x;
  });

  cv::Rect area_bound = cv::boundingRect(*nearest_area_it);

  double angle = compute_skew_correction_angle(threshold_image(area_bound));
  cv::Mat skew_matrix = make_skew_matrix(angle, area_bound.width * 0.5);

  cv::Mat area_image = image(area_bound * (1.0 / ratio));

  cv::Mat blured_area;
  cv::medianBlur(area_image, blured_area, 5);

  cv::Mat deskewed_area;
  cv::warpAffine(area_image, deskewed_area, skew_matrix, image.size());

  cv::Mat grayscale_area;
  cv::cvtColor(deskewed_area, grayscale_area, CV_RGB2GRAY);

  cv::Mat threshold_area;
  cv::threshold(grayscale_area, threshold_area, 128.0, 255.0, CV_THRESH_BINARY_INV);

  auto area_filled_areas = find_filled_areas(threshold_area.clone(), 255);
  auto max_area = std::max_element(area_filled_areas.begin(),
                                   area_filled_areas.end(),
                                   [&threshold_area](const std::vector<cv::Point> &a,
                                                     const std::vector<cv::Point> &b)
  {
    cv::Rect a_bound = cv::boundingRect(a);
    cv::Rect b_bound = cv::boundingRect(b);

    return a_bound.area() < b_bound.area();
  });
  draw_area(threshold_area, *max_area, 0);
  cv::imshow("threshold_area", threshold_area);

  tesseract::TessBaseAPI tess_api;
  tess_api.Init("tessdata", "eng");
  tess_api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
  tess_api.SetVariable("tessedit_char_whitelist", "1234567890");

  tess_api.SetImage(threshold_area.ptr(),
                    threshold_area.size().width,
                    threshold_area.size().height,
                    threshold_area.elemSize(),
                    threshold_area.step1());
  char *text = tess_api.GetUTF8Text();
  std::string place_number = text;
  delete[] text;

  place_number.erase(std::remove_if(place_number.begin(),
                                    place_number.end(),
                                    [](int char_value)
  {
    return !isdigit(char_value);
  }), place_number.end());

  return place_number;
}
