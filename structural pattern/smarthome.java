import static java.lang.System.out;

import java.util.ArrayList;
import java.util.List;


public class smarthome {
    public static void main(String[] args) {
          List<SmartDevice> devices = new ArrayList<>();
        devices.add(new SmartLight());
        devices.add(new SmartFan());
        devices.add(new oldadapter(new OldSmartBulb()));
        devices.add(new heateradapter(new LegacyHeater()));

        for (SmartDevice d : devices) {
            d.turnOn();
            d.turnOff();
        }
    }
}


interface SmartDevice { 
   void turnOn(); 
   void turnOff(); 
}

class SmartLight implements SmartDevice{
    public void turnOff(){
        out.println("Light turned off");
    }

    public void turnOn(){
                out.println("Light turned on");

    }
}



class SmartAC implements SmartDevice{
    public void turnOff(){
                out.println("ac turned off");

    }   

    public void turnOn(){
                out.println("ac turned on");

    }
}


class SmartFan implements SmartDevice{
    public void turnOff(){
    out.println("fan turned off");

    }

    public void turnOn(){
    out.println("fan turned on");

    }
}

class OldSmartBulb{ 
   public void powerOn() {} 
   public void powerOff() {} 
}
class LegacyHeater { 
   public void startHeating() {} 
   public void stopHeating() {} 
}


class oldadapter implements SmartDevice{
    protected OldSmartBulb Legacy;
    oldadapter(OldSmartBulb Legacy){
        this.Legacy=Legacy;
    }

    public void turnOff(){
       
            
            Legacy.powerOff();
        
    }

    public void turnOn(){
       
            
            Legacy.powerOn();
        
    }
}


class heateradapter implements SmartDevice{
    protected LegacyHeater Legacy;
    heateradapter(LegacyHeater Legacy){
        this.Legacy=Legacy;
    }

    public void turnOff(){
       
            
            Legacy.stopHeating();
        
    }

    public void turnOn(){
       
            
            Legacy.startHeating();
        
    }
}

