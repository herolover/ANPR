#include "place_number_recognizer.h"

#include "help_opencv.h"
#include "help_alg.h"

#include <algorithm>

#include <tesseract/baseapi.h>

std::pair<std::string, cv::Rect> recognize_place_number(const cv::Mat &image,
                                                        const Color &color,
                                                        const cv::Rect &search_rect)
{
//  double ratio = 0.4;
  cv::Mat small_image;
//  cv::resize(image, small_image, cv::Size(), ratio, ratio);
  small_image = image(search_rect);

  cv::Mat gray;
  cv::cvtColor(small_image, gray, CV_RGB2GRAY);

  cv::Mat blur;
  cv::medianBlur(gray, blur, 3);

  cv::Mat canny;
  cv::Canny(blur, canny, 80, 50);

//  cv::imshow("canny", canny);

  cv::Mat elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::Mat dilate_img;
  cv::dilate(canny, dilate_img, elem);

  cv::imshow("dilate_img", dilate_img);


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

  cv::Mat threshold_img = cv::Mat::zeros(dilate_img.size(), CV_8UC1);
  for (auto &area1: filled_areas)
  {
    for (auto &area2: filled_areas)
    {
      if (area1.second.contains(area2.second.tl()) &&
          area1.second.contains(area2.second.br()))
      {
        double ratio1 = (double)area1.second.width / area1.second.height;
        double ratio2 = (double)area2.second.width / area2.second.height;
        double ratio = ratio1 / ratio2;

        if (ratio > 0.9 && ratio < 1.1 &&
            (double)area1.second.area() / area2.second.area() < 2.0)
        {
          draw_area(threshold_img, area1.first, 255);
          break;
        }
      }
    }
//    std::cout << area.second << " " << is_rectangle(area.first) << std::endl;
  }
  cv::imshow("threshold_img", threshold_img);

  return std::make_pair("", cv::Rect());
//  double ratio = 0.4;
//  cv::Mat small_image;
//  cv::resize(image, small_image, cv::Size(), ratio, ratio);

//  auto minmax_color = find_min_max_in_rect_center<3>(image, search_rect);
//  cv::Mat balanced_image;
//  set_white_balance<3>(small_image, balanced_image,
//                       minmax_color.first, minmax_color.second);
////  cv::imshow("balanced_image", balanced_image);

//  cv::Mat difference_image(balanced_image.size(), CV_8UC1);
//  for (int i = 0; i < balanced_image.cols; ++i)
//    for (int j = 0; j < balanced_image.rows; ++j)
//    {
//      cv::Vec<int, 3> diff = (cv::Vec<int, 3>)balanced_image.at<Color>(j, i) - (cv::Vec<int, 3>)color;
//      double diff_norm = sqr(diff(0)) + sqr(diff(1)) + sqr(diff(2));

//      double value = 255.0 * sqrt(diff_norm / (255.0 * 255.0 + 255.0 * 255.0 + 255.0 * 255.0));

//      difference_image.at<unsigned char>(j, i) = (unsigned char)value;
//    }

////  cv::imshow("difference_image", difference_image);

//  cv::Mat threshold_image;
//  cv::threshold(difference_image, threshold_image, 80.0, 255.0, CV_THRESH_BINARY_INV);

////  cv::imshow("threshold_image", threshold_image);

//  auto filled_areas = find_filled_areas(threshold_image.clone(), 255);

//  double image_center_x = threshold_image.cols * 0.5;
//  filled_areas.erase(std::remove_if(filled_areas.begin(), filled_areas.end(),
//                                    [&image_center_x](const std::vector<cv::Point> &area)
//  {
//    cv::Rect area_bound = cv::boundingRect(area);

//    return area_bound.x > image_center_x || area.size() < 40;
//  }), filled_areas.end());

//  if (filled_areas.size() == 0)
//  {
//    return "";
//  }

//  auto place_number_area = std::max_element(filled_areas.begin(), filled_areas.end(),
//                                            [](const std::vector<cv::Point> &a,
//                                               const std::vector<cv::Point> &b)
//  {
//    return is_rectangle(a) < is_rectangle(b);
//  });

//  cv::Rect area_bound = cv::boundingRect(*place_number_area);
//  int w = area_bound.width / 12;
//  int h = area_bound.height / 12;
//  area_bound.x += w;
//  area_bound.y += h;
//  area_bound.width -= 2 * w;
//  area_bound.height -= 2 * h;

//  double angle = compute_skew_correction_angle(threshold_image(area_bound), 150);
//  cv::Mat skew_matrix = make_skew_matrix(angle, area_bound.width * 0.5);

//  cv::Mat area_image = image(area_bound * (1.0 / ratio));

////  cv::Mat blured_image;
////  cv::medianBlur(area_image, blured_image, 3);

//  cv::Mat deskewed_area;
//  cv::warpAffine(area_image, deskewed_area, skew_matrix, area_image.size());

//  cv::Mat grayscale_area;
//  cv::cvtColor(deskewed_area, grayscale_area, CV_RGB2GRAY);

////  cv::imshow("grayscale_area", grayscale_area);

//  cv::Mat threshold_area;
//  cv::threshold(grayscale_area, threshold_area, 115.0, 255.0, CV_THRESH_BINARY_INV);

//  auto area_filled_areas = find_filled_areas(threshold_area.clone(), 255);
//  auto max_area = std::max_element(area_filled_areas.begin(),
//                                   area_filled_areas.end(),
//                                   [&threshold_area](const std::vector<cv::Point> &a,
//                                                     const std::vector<cv::Point> &b)
//  {
//    cv::Rect a_bound = cv::boundingRect(a);
//    cv::Rect b_bound = cv::boundingRect(b);

//    return a_bound.area() < b_bound.area();
//  });

////  cv::imshow("threshold_area_before", threshold_area);
//  draw_area(threshold_area, *max_area, 0);

////  cv::imshow("threshold_area", threshold_area);

//  tesseract::TessBaseAPI tess_api;
//  tess_api.Init("tessdata", "eng");
//  tess_api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
//  tess_api.SetVariable("tessedit_char_whitelist", "1234567890");

//  tess_api.SetImage(threshold_area.ptr(),
//                    threshold_area.size().width,
//                    threshold_area.size().height,
//                    threshold_area.elemSize(),
//                    threshold_area.step1());
//  char *text = tess_api.GetUTF8Text();
//  std::string place_number = text;
//  delete[] text;

//  place_number.erase(std::remove_if(place_number.begin(),
//                                    place_number.end(),
//                                    [](int char_value)
//  {
//    return !isdigit(char_value);
//  }), place_number.end());

//  return place_number;
}
