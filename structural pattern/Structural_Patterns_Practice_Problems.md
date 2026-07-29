# Structural Design Pattern — Practice Problems
### (Decorator, Adapter, Bridge)

Each problem follows the same format as the original: a scenario, your task, a **provided** class you must use as-is, a `Main` class showing how it's exercised, and the exact expected console output.

---

## Problem 1 — Decorator Pattern

**Duration:** 35 minutes

A café originally sold only plain beverages (e.g., Espresso, Tea) at fixed prices. As the business grew, customers began asking to customize their drinks with **add-ons** such as Milk, Sugar, Caramel Syrup, and Whipped Cream — and often more than one of the same add-on (e.g., double caramel).

Each add-on increases both the **price** and the **description** of the beverage. New add-ons (e.g., Vanilla, Extra Shot) should be introducible in the future **without modifying** the existing beverage or add-on classes.

### Your Task
Implement the system so it can:
1. Build a beverage by wrapping it with any number/combination of add-ons, in any order.
2. Print a receipt showing the full description (base drink + all add-ons, in the order applied) and the final price of each beverage.
3. Calculate the total bill across all beverages in an order.

The `CafeOrder` class below is provided for your convenience — it treats every beverage (customized or not) uniformly as a single `Beverage`. Your implementation must work with it unchanged.

```java
import java.util.ArrayList;
import java.util.List;

public class CafeOrder {
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

public class Main {
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
```

**Base prices:** Espresso = £2.50, Tea = £2.00
**Add-on prices:** Milk = £0.50, Sugar = £0.20, Caramel = £0.70, Whipped Cream = £0.80

### Expected Output
```
========== CAFE RECEIPT ==========
Tea, Milk, Sugar - £2.70
Espresso, Caramel, Whipped Cream, Caramel - £4.70
-----------------------------
Total Bill: £7.40
```

---

## Problem 2 — Adapter Pattern

**Duration:** 35 minutes

A travel booking startup wants to offer flight search and booking through its own unified interface. However, the actual flights come from two **incompatible third-party systems** it has already signed contracts with:

- A legacy **XML-based** airline system with methods like `fetchFlightsXML(...)` and `reserveSeatXML(...)`.
- A newer **JSON-based** airline system with methods like `getAvailableFlightsJSON(...)` and `confirmBookingJSON(...)`.

Neither legacy system can be modified (they're external libraries), yet the platform must search and book flights from both **uniformly**, and it must be easy to plug in a third provider later without touching existing code.

### Your Task
1. Define a common booking interface that the rest of the system relies on.
2. Make both legacy systems work through that common interface without changing their source code.
3. Search and book a flight through each provider uniformly, printing results in a consistent format.

The `TravelAgency` class below is provided for your convenience — it holds a simple list of booking-capable providers and treats them all the same way.

```java
import java.util.ArrayList;
import java.util.List;

public class TravelAgency {
    private List<BookingService> providers = new ArrayList<>();

    public void add(BookingService provider) {
        providers.add(provider);
    }

    public void searchAndBookAll(String origin, String destination) {
        System.out.println("========== FLIGHT SEARCH ==========");
        for (BookingService provider : providers) {
            System.out.println("Provider: " + provider.getClass().getSimpleName());
            List<String> flights = provider.searchFlights(origin, destination);
            System.out.println(" Available Flights: " + flights);
            String chosen = flights.get(0);
            System.out.println(" Booking Flight: " + chosen);
            System.out.println(" " + provider.bookFlight(chosen));
        }
        System.out.println("-----------------------------");
        System.out.println("All bookings completed.");
    }
}

// --- Existing legacy systems (cannot be modified) ---
class LegacyFlightSystemXML {
    public String[] fetchFlightsXML(String origin, String destination) {
        return new String[]{"XML-101", "XML-202"};
    }
    public String reserveSeatXML(String flightCode) {
        return "Reserved via XML system: " + flightCode;
    }
}

class LegacyFlightSystemJSON {
    public List<String> getAvailableFlightsJSON(String origin, String destination) {
        return java.util.Arrays.asList("JSON-501", "JSON-502");
    }
    public String confirmBookingJSON(String flightCode) {
        return "Booking confirmed [JSON]: " + flightCode;
    }
}

public class Main {
    public static void main(String[] args) {
        TravelAgency agency = new TravelAgency();
        agency.add(new FlightAdapterXML(new LegacyFlightSystemXML()));
        agency.add(new FlightAdapterJSON(new LegacyFlightSystemJSON()));

        agency.searchAndBookAll("Dhaka", "London");
    }
}
```

### Expected Output
```
========== FLIGHT SEARCH ==========
Provider: FlightAdapterXML
 Available Flights: [XML-101, XML-202]
 Booking Flight: XML-101
 Reserved via XML system: XML-101
Provider: FlightAdapterJSON
 Available Flights: [JSON-501, JSON-502]
 Booking Flight: JSON-501
 Booking confirmed [JSON]: JSON-501
-----------------------------
All bookings completed.
```

---

## Problem 3 — Bridge Pattern

**Duration:** 35 minutes

A smart home company sells devices (currently **Lights** and **Fans**) that can each be controlled through different **communication protocols** (currently **WiFi** and **Bluetooth**). Every combination of device type and protocol must work — a Light over WiFi, a Fan over Bluetooth, and so on — and the company plans to add more device types (e.g., Thermostat) and more protocols (e.g., Zigbee) in the future, independently of each other, without an explosion of subclasses like `WiFiLight`, `BluetoothLight`, `WiFiFan`, `BluetoothFan`, etc.

### Your Task
1. Separate the *device* hierarchy from the *communication protocol* hierarchy so each can vary independently.
2. Every device must be able to turn on and adjust a "level" (brightness for a Light, speed for a Fan), regardless of which protocol it uses.
3. Turn on and adjust all devices in a hub uniformly, printing a consistent status report.

The `SmartHomeHub` class below is provided for your convenience — it holds a simple list of devices and controls them all the same way.

```java
import java.util.ArrayList;
import java.util.List;

public class SmartHomeHub {
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

public class Main {
    public static void main(String[] args) {
        Device livingRoomLight = new Light("Living Room Light", new WiFiController());
        Device bedroomFan = new Fan("Bedroom Fan", new BluetoothController());

        SmartHomeHub hub = new SmartHomeHub();
        hub.add(livingRoomLight);
        hub.add(bedroomFan);

        hub.report(75);
    }
}
```

### Expected Output
```
========== SMART HOME HUB ==========
Light 'Living Room Light' turning ON via WiFi
  [WiFi] Signal sent: Power ON
Fan 'Bedroom Fan' turning ON via Bluetooth
  [Bluetooth] Signal sent: Power ON
Setting Living Room Light brightness to 75%
  [WiFi] Signal sent: Set level to 75
Setting Bedroom Fan speed to 75%
  [Bluetooth] Signal sent: Set level to 75
-----------------------------
All devices updated.
```
