import static java.lang.System.out;

public class decorator2 {
    public static void main(String[] args) {
        Rmdpackage p=new standardpackage();
        p=new gift(p);
        p=new fruit(p);
        p=new sweet(p);
        p.makepackage("Order for Mr rahim");
        out.println("\n Total Price:"+p.getprice());
        
    }
}
interface Rmdpackage{
    void makepackage(String p);
    double getprice();
}

class standardpackage implements Rmdpackage{
     public void makepackage(String p){
        out.println(" standard package --->"+p);
     };
     public double getprice(){
        return 50;
     }
}

class specialpackage implements Rmdpackage{
    public void makepackage(String p){
        out.println(" special package --->"+p);
     };
     public double getprice(){
        return 100;
     }
}

class premiumpackage implements Rmdpackage{
    public void makepackage(String p){
        out.println(" premium package --->"+p);
     };
      public double getprice(){
        return 150;
     }
}

class packagedecorator implements Rmdpackage{
    protected Rmdpackage wrappee;
    public packagedecorator(Rmdpackage wrappee){
        this.wrappee=wrappee;
    }
    public void makepackage(String p){
       wrappee.makepackage(p);
     };
     public double getprice(){
        return wrappee.getprice();
     }
}


class fruit extends packagedecorator{
    public fruit(Rmdpackage wrappee){
        super(wrappee);
    }

   public void makepackage(String p){
       p= "(Fruit Added on)" +p;
       super.makepackage(p);
     };
     public double getprice(){
       return super.getprice()+5;
     }
}
class sweet extends packagedecorator{
    public sweet(Rmdpackage wrappee){
        super(wrappee);
    }
     public void makepackage(String p){
      p= "(sweet Added on)" +p;
       super.makepackage(p);
     };
     public double getprice(){
       return super.getprice()+5;
     }
}

class gift extends packagedecorator{
    public gift(Rmdpackage wrappee){
        super(wrappee);
    }

    public void makepackage(String p){
      p= "(gift Added on)" +p;
       super.makepackage(p);
     };
     public double getprice(){
       return super.getprice()+5;
     }
}


