#include "place_number_recognizer.h"


void PlaceNumberRecognizer::find_and_recognize()
{
  double ratio = 0.5;
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

  cv::imshow("difference_image", difference_image);

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
}


void PlaceNumberRecognizer::set_image(const cv::Mat &image)
{
  this->image_ = image;
}


void PlaceNumberRecognizer::set_color(const Color &color)
{
  this->color_ = color;
}
