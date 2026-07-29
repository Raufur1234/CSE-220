import java.util.ArrayList;
import java.util.List;
import static java.lang.System.out;

public class Main {
 public static void main(String[] args) {
 // Foods
 Food burger = new Food("Burger", 8);
 Food pizza = new Food("Pizza", 10);
 Food fries = new Food("French Fries", 3);
 // Set Menu
 SetMenu lunch = new SetMenu("Lunch Combo");
 lunch.addFood(burger);
 lunch.addFood(fries);
 // Grocery Items
 Grocery rice = new Grocery("Rice", 20);
 Grocery oil = new Grocery("Cooking Oil", 12);
 Grocery eggs = new Grocery("Eggs", 6);
 Grocery sugar = new Grocery("Sugar", 5);
 // Small Package
 GroceryPackage breakfastPack = new GroceryPackage("Breakfast Pack");
 breakfastPack.add(eggs);
 breakfastPack.add(sugar);
 // Large Package (contains another package)
 GroceryPackage monthlyPack = new GroceryPackage("Monthly Essentials");
 monthlyPack.add(rice);
 monthlyPack.add(oil);
 monthlyPack.add(breakfastPack);
 
 // Customer Order
 Order order = new Order();
 order.add(pizza);
 order.add(lunch);
 order.add(rice);
 order.add(monthlyPack);
 order.printReceipt();
 }
}
interface OrderItem{
double getPrice();
void print(String indent); 
}
class Food implements OrderItem{
public String Name;
public double Price;
Food(String Name,double Price){
this.Name=Name;
this.Price=Price;
}
public double getPrice(){
    return Price;
}

public void print(String indent){
    System.out.println(indent+" Food: "+Name+"(£"+Price+")");
}
}

class SetMenu implements OrderItem{
private String name;
private List<Food> items = new ArrayList<>();
public SetMenu(String Name){
    name=Name;
}

public void addFood(Food item) {
    items.add(item);
}
public double getPrice(){
    double total=0;
    for (Food item:items){total += item.getPrice();}
    return total;
}

public void print(String indent){
    out.println("\nSet Menu:"+ name);
    for (Food item:items){item.print(indent);}
   
}

}

interface gc extends OrderItem{}

class Grocery implements gc{
private String name;
private double price;
public Grocery(String Name,double Price){
name=Name;
price=Price;
}
public double getPrice(){
    return price;
}
public void print(String indent){
    System.out.println(indent+" Grocery: "+name+"(£"+price+")");
}

}

class GroceryPackage implements gc{
private String name;
private List<gc> items = new ArrayList<>();
public GroceryPackage(String Name){
    name=Name;
}

public void add(gc item) {
    items.add(item);
}
public double getPrice(){
    double total=0;
    for (gc item:items){total += item.getPrice();}
    return total;
}

public void print(String indent){
    out.println("\nPackage:"+ name);
    for (gc item:items){item.print(indent);}

}


}

class Order {
 private List<OrderItem> items = new ArrayList<>();
 public void add(OrderItem item) {
 items.add(item);
 }
 public double getTotalPrice() {
 double total = 0;
 for (OrderItem item : items) {
 total += item.getPrice();
 }
 return total;
 }
 public void printReceipt() {
 System.out.println("========== RECEIPT ==========");
 for (OrderItem item : items) {
 item.print("");
 }
 System.out.println("-----------------------------");
 System.out.printf("Total Bill: £%.2f%n", getTotalPrice());
 }
}