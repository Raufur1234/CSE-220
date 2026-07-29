import java.util.ArrayList;
import java.util.List;

class SmartHomeHub {
    private List<Device> devices = new ArrayList<>();

    public void add(Device device) {
        devices.add(device);
    }

    public void turnOnAll() {
        for (Device d : devices) {
            d.turnOn();
        }
    }

    public void adjustAll(int level) {
        for (Device d : devices) {
            d.adjust(level);
        }
    }

    public void report(int level) {
        System.out.println("========== SMART HOME HUB ==========");
        turnOnAll();
        adjustAll(level);
        System.out.println("-----------------------------");
        System.out.println("All devices updated.");
    }
}

public class bridge {
    public static void main(String[] args) {
        Device livingRoomLight = new Light("Living Room Light", new WiFiController());
        Device bedroomFan = new Fan("Bedroom Fan", new BluetoothController());

        SmartHomeHub hub = new SmartHomeHub();
        hub.add(livingRoomLight);
        hub.add(bedroomFan);

        hub.report(75);
    }
}

interface Device{
    void turnOn();
    void turnOff();
    void adjust(double level);

}

interface Controller{
String send_signal();
}

class WiFiController implements Controller{
    public WiFiController(){

    }
    public String send_signal(){
        return "WiFi";
    }
}

class BluetoothController implements Controller{
    public BluetoothController(){

    }
    public String send_signal(){
        return "Bluetooth";
    }
}

class Light implements Device{
    String name;
    protected Controller c;
    Light(String name,Controller c){
        this.name=name;
        this.c=c;
    }

    public void turnOn(){
        System.out.println("Light " + this.name+" turning on via"+c.send_signal());
         System.out.println("\n ["+c.send_signal()+"]Signal sent: Power ON");
    }
     public void turnOff(){
        System.out.println("Light" + this.name+"turning off via"+c.send_signal());
         System.out.println("\n ["+c.send_signal()+"]Signal sent: Power OFF");
    }
      public void adjust(double level){
        System.out.println("Setting " + this.name+" brightness to"+level+"%" );
         System.out.println("\n ["+c.send_signal()+"]Signal sent: Set level to"+level+"%");
    }
}


class Fan implements Device{
    String name;
    protected Controller c;
    Fan(String name,Controller c){
        this.name=name;
        this.c=c;
    }

    public void turnOn(){
        System.out.println("Fan " + this.name+" turning on via"+c.send_signal());
         System.out.println("\n ["+c.send_signal()+"]Signal sent: Power ON");
    }
     public void turnOff(){
        System.out.println("Fan " + this.name+" turning off via"+c.send_signal());
         System.out.println("\n ["+c.send_signal()+"]Signal sent: Power OFF");
    }
      public void adjust(double level){
        System.out.println("Setting " + this.name+" Fan speed to"+level+"%" );
         System.out.println("\n ["+c.send_signal()+"]Signal sent: Set level to"+level+"%");
    }
}
// ========== SMART HOME HUB ==========
// Light 'Living Room Light' turning ON via WiFi
//   [WiFi] Signal sent: Power ON
// Fan 'Bedroom Fan' turning ON via Bluetooth
//   [Bluetooth] Signal sent: Power ON
// Setting Living Room Light brightness to 75%
//   [WiFi] Signal sent: Set level to 75
// Setting Bedroom Fan speed to 75%
//   [Bluetooth] Signal sent: Set level to 75
// -----------------------------
// All devices updated.

