#include "place_number_recognizer.h"

#include "help_opencv.h"

#include <algorithm>

#include <tesseract/baseapi.h>


void PlaceNumberRecognizer::find_and_recognize()
{
  double ratio = 0.3;
  cv::Mat small_image;
  cv::resize(this->image_, small_image, cv::Size(), ratio, ratio);

  cv::Mat blured_image;
  cv::medianBlur(small_image, blured_image, 5);

  Color nearest_color = *std::min_element(blured_image.begin<Color>(),
                                          blured_image.end<Color>(),
                                          [this](const Color &a, const Color &b)
  {
    Color diff_a;
    cv::absdiff(a, this->color_, diff_a);

    Color diff_b;
    cv::absdiff(b, this->color_, diff_b);

    return cv::norm(diff_a) < cv::norm(diff_b);
  });

  cv::Mat difference_image(blured_image.size(), blured_image.type());
  for (int i = 0; i < blured_image.cols; ++i)
    for (int j = 0; j < blured_image.rows; ++j)
    {
      cv::Vec<int, 3> color = blured_image.at<Color>(j, i);

      color -= (cv::Vec<int, 3>)nearest_color;
      color[0] = std::fabs(color[0]);
      color[1] = std::fabs(color[1]);
      color[2] = std::fabs(color[2]);
      if (color[0] > 255)
        color[0] = 255;
      if (color[1] > 255)
        color[1] = 255;
      if (color[2] > 255)
        color[2] = 255;

      difference_image.at<Color>(j, i) = (Color)color;
    }

  const double threshold_coefficient = 20.0;
  cv::Mat threshold_image(difference_image.size(), CV_8UC1);
  for (int i = 0; i < difference_image.rows; ++i)
    for (int j = 0; j < difference_image.cols; ++j)
    {
      Color color = difference_image.at<Color>(i, j);

      unsigned char value = 0;
      if (cv::norm(color) < threshold_coefficient)
        value = 255;

      threshold_image.at<unsigned char>(i, j) = value;
    }

  cv::imshow("threshold_image", threshold_image);

  auto filled_areas = find_filled_areas(threshold_image.clone(), 255);

  std::sort(filled_areas.begin(), filled_areas.end(),
            [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
  {
    return a.size() > b.size();
  });

  filled_areas.erase(std::remove_if(filled_areas.begin(), filled_areas.end(),
                                    [](const std::vector<cv::Point> &area)
  {
    cv::Rect area_bound = cv::boundingRect(area);

    return area_bound.width / (double)area_bound.height > 2 ||
           area_bound.height / (double)area_bound.width > 2 || area.size() < 60;
  }), filled_areas.end());

  for (auto &area: filled_areas)
  {
    cv::Rect area_bound = cv::boundingRect(area);

    double angle = compute_skew_correction_angle(threshold_image(area_bound));
    cv::Mat skew_matrix = make_skew_matrix(angle, area_bound.width * 0.5);

    cv::Mat area_image = this->image_(area_bound * (1.0 / ratio));

    cv::Mat blured_area;
    cv::medianBlur(area_image, blured_area, 5);

    cv::Mat deskewed_area;
    cv::warpAffine(area_image, deskewed_area, skew_matrix, this->image_.size());

    cv::Mat grayscale_area;
    cv::cvtColor(deskewed_area, grayscale_area, CV_RGB2GRAY);

    cv::Mat threshold_area;
    cv::threshold(grayscale_area, threshold_area, 128.0, 255.0, CV_THRESH_BINARY_INV);

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
    this->place_number_ = text;
    delete[] text;

    this->place_number_.erase(std::remove_if(this->place_number_.begin(),
                                             this->place_number_.end(),
                                             [](int char_value)
    {
      return !isdigit(char_value);
    }), this->place_number_.end());

    std::cout << "Place number: " << this->place_number_ << std::endl;
  }
}


void PlaceNumberRecognizer::set_image(const cv::Mat &image)
{
  this->image_ = image;
}


void PlaceNumberRecognizer::set_color(const Color &color)
{
  this->color_ = color;
}
