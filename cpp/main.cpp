#include <iostream>
#include <string>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <utility>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "ANPR.h"
#include "place_number_recognizer.h"
#include "help_opencv.h"


std::string folder;
int max_image_index = 0;
int image_index = 0;
cv::Point2i search_rect_center;
cv::Rect search_rect;
cv::Mat image;
cv::Mat small_image;
double ratio = 1.0;

void on_mouse(int event, int x, int y, int, void *)
{
  static bool capture = false;

  if (event == CV_EVENT_LBUTTONDOWN)
  {
    search_rect_center.x = x;
    search_rect_center.y = y;
    capture = true;
  }

  if (capture)
  {
    int half_width = std::abs(x - search_rect_center.x);
    int half_height = std::abs(y - search_rect_center.y);
    search_rect.width = half_width * 2;
    search_rect.height = half_height * 2;
    search_rect.x = search_rect_center.x - half_width;
    search_rect.y = search_rect_center.y - half_height;

    cv::Mat tmp_smal_image = small_image.clone();
    cv::rectangle(tmp_smal_image, search_rect, cv::Scalar(0, 0, 255), 2);
    cv::imshow("image", tmp_smal_image);
  }

  if (event == CV_EVENT_LBUTTONUP)
    capture = false;
}

void load_image()
{
  std::cout << folder + boost::str(boost::format("%03d.jpg") % image_index) << std::endl;
  image = cv::imread(folder + boost::str(boost::format("%03d.jpg") % image_index));

  ratio = 0.3;

  cv::resize(image, small_image, cv::Size(), ratio, ratio);

  cv::imshow("image", small_image);
}

void left_key_handler()
{
  image_index += 1;
  if (image_index >= max_image_index)
    image_index = 0;

  load_image();
}

void right_key_handler()
{
  image_index -= 1;
  if (image_index < 0)
    image_index = max_image_index - 1;

  load_image();
}

void process()
{
//  std::cout << ANPR::recognize_number_plate(image, search_rect * (1.0 / ratio)).first << std::endl;
  std::cout << recognize_place_number(image, Color(0, 0, 196), search_rect * (1.0 / ratio)).first << std::endl;
}

int main(int argc, char *argv[])
{
  folder += std::string(argv[1]) + "/";
  boost::filesystem::path path(folder);
  max_image_index = std::distance(boost::filesystem::directory_iterator(path),
                                  boost::filesystem::directory_iterator());

  cv::namedWindow("image");
  cv::setMouseCallback("image", on_mouse);

  load_image();

  enum KEY
  {
    ESCAPE_KEY = 27,
    LEFT_KEY = 65363,
    RIGHT_KEY = 65361,
    SPACE_KEY = 32
  };

  std::unordered_map<int, std::function<void ()>> key_handlers =
  {
    {LEFT_KEY, left_key_handler},
    {RIGHT_KEY, right_key_handler},
    {SPACE_KEY, process}
  };

  KEY key;
  while ((key = (KEY)cv::waitKey()) != ESCAPE_KEY)
  {
    if (key_handlers.count(key) == 1)
      key_handlers[key]();
  }

  return 0;
}
