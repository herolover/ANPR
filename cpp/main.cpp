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


void left_key_handler();
void right_key_handler();
void process();


const std::string folder = "../../test_img/my/";

boost::filesystem::path path(folder);
int max_image_index = std::distance(boost::filesystem::directory_iterator(path),
                                    boost::filesystem::directory_iterator());

int image_index = 5;


enum KEY
{
  ESCAPE_KEY = 27,
  LEFT_KEY = 65363,
  RIGHT_KEY = 65361
};

int main()
{
  process();

  std::map<KEY, std::function<void ()>> key_handlers =
  {
    {LEFT_KEY, left_key_handler},
    {RIGHT_KEY, right_key_handler}
  };

  KEY key;
  while ((key = (KEY)cv::waitKey()) != ESCAPE_KEY)
  {
    if (key_handlers.count(key) == 1)
      key_handlers[key]();
  }

  return 0;
}


void left_key_handler()
{
  image_index += 1;
  if (image_index >= max_image_index)
    image_index = 0;

  process();
}


void right_key_handler()
{
  image_index -= 1;
  if (image_index < 0)
    image_index = max_image_index - 1;

  process();
}


void process()
{
  std::cout << folder + boost::str(boost::format("%03d.jpg") % image_index) << std::endl;
  cv::Mat image = cv::imread(folder + boost::str(boost::format("%03d.jpg") % image_index));

  ANPR anpr;
  anpr.set_image(image);

  anpr.run();

  cv::Mat number_plate_image = anpr.get_number_plate_image();

  if (!number_plate_image.empty())
  {
    std::cout << "Text: " << anpr.get_number_plate_text() << std::endl;
    cv::imshow("number plate image", number_plate_image);
  }
}
