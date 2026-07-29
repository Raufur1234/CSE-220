import java.util.ArrayList;
import java.util.List;
import static java.lang.System.out;

// ---------- Component ----------
interface BazarItem {
    double getPrice();
    double getWeight();
    String describe(String indent);
}

// ---------- Leaf ----------
class SingleItem implements BazarItem {
    private String name;
    private double price;
    private double weight;

    SingleItem(String name, double price, double weight) {
        this.name = name;
        this.price = price;
        this.weight = weight;
    }

    public double getPrice()  { return price; }
    public double getWeight() { return weight; }

    public String describe(String indent) {
        return indent + name + " (Tk " + price + ", " + weight + "kg)\n";
    }
}

// ---------- Composite ----------
class Package implements BazarItem {
    private String name;
    private List<BazarItem> contents = new ArrayList<>();

    Package(String name) { this.name = name; }

    void add(BazarItem item)    { contents.add(item); }
    void remove(BazarItem item) { contents.remove(item); }

    public double getPrice() {
        double total = 0;
        for (BazarItem item : contents) total += item.getPrice();
        return total;
    }

    public double getWeight() {
        double total = 0;
        for (BazarItem item : contents) total += item.getWeight();
        return total;
    }

    public String describe(String indent) {
        StringBuilder sb = new StringBuilder(indent + name + ":\n");
        for (BazarItem item : contents) {
            sb.append(item.describe(indent + "  "));
        }
        return sb.toString();
    }
}

public class composite1 {
    public static void main(String[] args) {
        // Single items
        SingleItem rice = new SingleItem("Rice", 60, 5);
        SingleItem oil  = new SingleItem("Oil", 180, 2);
        SingleItem pulse = new SingleItem("Pulse", 90, 1);

        // Preset package: Small = rice + oil
        Package small = new Package("Small Package");
        small.add(rice);
        small.add(oil);

        // Custom bundle: Small preset + extra pulse + loose item
        Package custom = new Package("Custom Bazar for Mr Rahim");
        custom.add(small);
        custom.add(pulse);
        custom.add(new SingleItem("Sugar", 70, 1));

        out.println(custom.describe(""));
        out.println("Total price: " + custom.getPrice());
        out.println("Total weight: " + custom.getWeight() + "kg");
    }
}