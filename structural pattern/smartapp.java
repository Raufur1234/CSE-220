import static java.lang.System.out;

interface notifications{
    void notifys(String message);

}

class emailnotification implements notifications{
    public emailnotification(){

    }
    public void notifys(String message){
       out.println("notify via email:"+message);
    }
}

class smsnotification implements notifications{
    public smsnotification(){

    }
    public void notifys(String message){
       out.println("notify via sms"+message);
    }
}


class pushnotification implements notifications{
    public pushnotification(){

    }
   public void notifys(String message){
       out.println("notify via push"+message);
    }
}


class notificationdecorator implements notifications{
    protected notifications wrappee;
    public notificationdecorator(notifications wrappee){
        this.wrappee=wrappee;
    }
   
    
    public void notifys(String message){
       this.wrappee.notifys(message);
    }
}


class encryption extends notificationdecorator{

    public encryption(notifications wrappee){
        super(wrappee);
    }

    public String encrypt(String data){
        return new StringBuilder(data).reverse().toString();
    }
    public void notifys(String message){
        String encrypted=encrypt(message);
        super.notifys(encrypted);
    }
}

class priority extends notificationdecorator{

    public priority(notifications wrappee){
        super(wrappee);
    }

    public String prioritise(String data){
        return data+ " High priority";
    }
    public void notifys(String message){
        String vip=prioritise(message);
        super.notifys(vip);
    }
}


class logger extends notificationdecorator{

    public logger(notifications wrappee){
        super(wrappee);
    }

    public String log(String data){
        
        return "LOG:"+data;
    }
    public void notifys(String message){
        String logged=log(message);
        super.notifys(logged);
    }
}


public class smartapp{
    public static void main(String[] args) {
       
        notifications n = new emailnotification();
        n = new logger(n);
        n = new priority(n);
        n = new encryption(n);
        n.notifys("Hello world");  
    }
}

