zx`135y import static java.lang.System.out;

public class bridge2 {
    public static void main(String[] args) {
    deliverydevice bike = new basicdelivery();
    deliverydevice drone = new dronedelivery();
    deliverydevice robot = new robotdelivery();

   new standard_delivery(bike).description("Order A");
   new scheduled_delivery(drone, "6 PM - 8 PM").description("Order B");
   new express_delivery(robot).description("Order C");
    }
    
}

interface deliverydevice{
    double Price(double price);
    double Time(double time);
    String route();
}

class basicdelivery implements deliverydevice{
    public double Price(double price){
        return price;
    }
    public double Time(double time){
        return time;
    }

    public String route(){
        return "routed via standard route";
    }
}
class dronedelivery implements deliverydevice{
    public double Price(double price){
        return price+10;
    }
    public double Time(double time){
        return time/2;
    }

    public String route(){
        return "routed via drone: ";
    }
}

class robotdelivery implements deliverydevice{
    public double Price(double price){
        return price+5;
    }
    public double Time(double time){
        return time*0.75;
    }

    public String route(){
        return "routed via robot: ";
    }
}


abstract class deliveryservice{
    protected deliverydevice device;
    protected String time_slot;
    deliveryservice(deliverydevice device){
        this.device=device;
    }
       deliveryservice(deliverydevice device,String time_slot){
        this.device=device;
        this.time_slot=time_slot;
    }
    public abstract double totaltime();
    public abstract double totalprice();
    public abstract void description(String desc);
}


class standard_delivery extends deliveryservice{
    standard_delivery(deliverydevice device){
        super(device);
    }
     public double totaltime(){
        return device.Time(24);
     };
     public double totalprice(){
        return device.Price(100);
     }

     public void description(String desc){
      out.println("STANDARD DELIVERY(IN 24h OR LESS):"+device.route()+desc);
     }
}



class express_delivery extends deliveryservice{
    express_delivery(deliverydevice device){
        super(device);
    }
     public double totaltime(){
        return device.Time(4);
     };
     public double totalprice(){
        return device.Price(500);
     }

     public void description(String desc){
        out.println("EXPRESS DELIVERY(IN 4h OR LESS):"+device.route()+desc);
     }
}



class scheduled_delivery extends deliveryservice{


    scheduled_delivery(deliverydevice device,String time_slot){
        super(device,time_slot);
    }
     public double totaltime(){
        return device.Time(24);
     };
     public double totalprice(){
        return device.Price(100);
     }

     public void description(String desc){
        System.out.println("SCHEDULED DELIVERY AT:"+time_slot+device.route()+desc) ;
     }
}