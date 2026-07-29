import static java.lang.System.out;

public class Dalchal {
    public static void main(String[] args) {
        channel smsChannel=new SMS();
        channel emailChannel=new email();
        
        notification Notification1=new PaymentFailed(smsChannel);
        notification Notification2=new PaymentFailed(emailChannel);
        notification Notification3=new bazarconfirmed(emailChannel);

        Notification1.send("Hello world");
        Notification2.send("Hello world");
        Notification3.send("Hello world");

    }  
}

interface channel{
    String routemessage(String message);
}

class SMS implements channel{
    public String routemessage(String message){return "routed via SMS:"+message;};
}

class email implements channel{
    public String routemessage(String message){return "routed via email:"+message;};
}

abstract class notification{
    protected channel myChannel;
    protected notification(channel newchannel){
        myChannel=newchannel;
    }

    abstract public void send(String message);
}


class PaymentFailed extends notification{
    PaymentFailed(channel newChannel){super(newChannel);}

    public void send(String message){
        out.println("SENT NOTIFICATION"+myChannel.routemessage(":PAYMENT FAILED:"+message));
    }
}



class  ontheway extends notification{
    ontheway(channel newChannel){super(newChannel);}

    public void send(String message){
        out.println("SENT NOTIFICATION:"+myChannel.routemessage("On the way:"+message));
    }
}

class  bazarconfirmed extends notification{
    bazarconfirmed(channel newChannel){super(newChannel);}

    public void send(String message){
        out.println("SENT NOTIFICATION:"+myChannel.routemessage("bazrconfirmed:"+message));
    }
}