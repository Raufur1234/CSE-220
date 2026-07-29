import java.util.ArrayList;
import java.util.List;
//Espresso = £2.50, Tea = £2.00 Add-on prices: Milk = £0.50, Sugar = £0.20, Caramel = £0.70, Whipped Cream = £0.80
class CafeOrder {
    private List<Beverage> items = new ArrayList<>();

    public void add(Beverage beverage) {
        items.add(beverage);
    }

    public double getTotalPrice() {
        double total = 0;
        for (Beverage b : items) {
            total += b.getCost();
        }
        return total;
    }

    public void printReceipt() {
        System.out.println("========== CAFE RECEIPT ==========");
        for (Beverage b : items) {
            System.out.printf("%s - £%.2f%n", b.getDescription(), b.getCost());
        }
        System.out.println("-----------------------------");
        System.out.printf("Total Bill: £%.2f%n", getTotalPrice());
    }
}

public class decorator3 {
    public static void main(String[] args) {
        Beverage order1 = new Tea();
        order1 = new Milk(order1);
        order1 = new Sugar(order1);

        Beverage order2 = new Espresso();
        order2 = new Caramel(order2);
        order2 = new WhippedCream(order2);
        order2 = new Caramel(order2); // double caramel

        CafeOrder cafeOrder = new CafeOrder();
        cafeOrder.add(order1);
        cafeOrder.add(order2);
        cafeOrder.printReceipt();
    }
}

interface Beverage{
    double getCost();
    String getDescription(); 
}

class Tea implements Beverage{
    public Tea() {};

    public double getCost(){
        return 2.00;
    }

    public String getDescription(){
        return "Tea";
    };
}

class Espresso implements Beverage{
    public Espresso() {};

    public double getCost(){
        return 2.50;
    }

    public String getDescription(){
        return "Espresso";
    };
}

class addondecorator implements Beverage{
    protected Beverage wrappee;
    protected addondecorator(Beverage wrappee){
        this.wrappee=wrappee;
    }
    public double getCost(){
        return wrappee.getCost();
    }
    public String getDescription(){
        return wrappee.getDescription();
    }
}


class Milk extends addondecorator{
    public Milk(Beverage wrappee){
        super(wrappee);
    }

    public double getCost(){
        return super.getCost()+0.50;
    }
    public String getDescription(){
        return wrappee.getDescription()+",Milk";
    }
}


class WhippedCream extends addondecorator{
    public WhippedCream(Beverage wrappee){
        super(wrappee);
    }

    public double getCost(){
        return super.getCost()+0.80;
    }
    public String getDescription(){
        return wrappee.getDescription()+",WhippedCream";
    }
}


class Caramel extends addondecorator{
    public Caramel(Beverage wrappee){
        super(wrappee);
    }

    public double getCost(){
        return super.getCost()+0.70;
    }
    public String getDescription(){
        return wrappee.getDescription()+",Caramel";
    }
}


class Sugar extends addondecorator{
    public Sugar(Beverage wrappee){
        super(wrappee);
    }

    public double getCost(){
        return super.getCost()+0.20;
    }
    public String getDescription(){
        return wrappee.getDescription()+",Sugar";
    }
}