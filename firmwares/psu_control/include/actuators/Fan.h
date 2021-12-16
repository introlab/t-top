#ifndef FAN_H
#define FAN_H

class Fan {
  int m_speed;

  public:
    Fan();
    void begin();

    void update(float celcius);
};

#endif
