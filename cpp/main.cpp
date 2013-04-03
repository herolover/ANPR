#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <functional>
#include <algorithm>
#include <utility>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "ANPR.h"
#include "help_opencv.h"


void on_mouse(int event, int x, int y, int, void *);
void left_key_handler();
void right_key_handler();
void load_image();
void process();


const std::string folder = "../../test_img/my/";

boost::filesystem::path path(folder);
int max_image_index = std::distance(boost::filesystem::directory_iterator(path),
                                    boost::filesystem::directory_iterator());

int image_index = 5;
cv::Rect search_rect;
cv::Mat image;
cv::Mat small_image;
double ratio = 1.0;


enum KEY
{
  ESCAPE_KEY = 27,
  LEFT_KEY = 65363,
  RIGHT_KEY = 65361,
  SPACE_KEY = 32
};

int main()
{
  cv::namedWindow("image");
  cv::setMouseCallback("image", on_mouse);

  load_image();

  std::map<KEY, std::function<void ()>> key_handlers =
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


void on_mouse(int event, int x, int y, int, void *)
{
  static bool capture = false;

  if (event == CV_EVENT_LBUTTONDOWN)
  {
    search_rect.x = x;
    search_rect.y = y;
    capture = true;

  }

  if (capture)
  {
    search_rect.width = x - search_rect.x;
    search_rect.height = y - search_rect.y;

    cv::Mat tmp_smal_image = small_image.clone();
    cv::rectangle(tmp_smal_image, search_rect, cv::Scalar(0, 0, 255), 2);
    cv::imshow("image", tmp_smal_image);
  }

  if (event == CV_EVENT_LBUTTONUP)
    capture = false;
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


void load_image()
{
  std::cout << folder + boost::str(boost::format("%03d.jpg") % image_index) << std::endl;
  image = cv::imread(folder + boost::str(boost::format("%03d.jpg") % image_index));

  ratio = 0.3;

  cv::resize(image, small_image, cv::Size(), ratio, ratio);

  cv::imshow("image", small_image);
}


void process()
{
  ANPR anpr;
  anpr.set_image(image, search_rect * (1.0 / ratio));

  anpr.find_and_recognize();

  cv::Mat number_plate_image = anpr.get_number_plate_image();

  if (!number_plate_image.empty())
  {
    std::cout << "Text: " << anpr.get_number_plate_text() << std::endl;
    cv::imshow("number plate image", number_plate_image);
  }
}
