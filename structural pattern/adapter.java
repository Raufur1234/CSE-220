import java.util.ArrayList;
import java.util.List;

class TravelAgency {
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

public class adapter{
    public static void main(String[] args) {
        TravelAgency agency = new TravelAgency();
        agency.add(new FlightAdapterXML(new LegacyFlightSystemXML()));
        agency.add(new FlightAdapterJSON(new LegacyFlightSystemJSON()));

        agency.searchAndBookAll("Dhaka", "London");
    }
}

interface BookingService{
    public List<String> searchFlights(String origin, String destination);
    public String bookFlight(String Flight);
}

class FlightAdapterXML implements BookingService{
    protected LegacyFlightSystemXML lc;
    public FlightAdapterXML(LegacyFlightSystemXML LC){
        lc=LC;
    }
    public List<String> searchFlights(String origin, String destination){
        return java.util.Arrays.asList(lc.fetchFlightsXML( origin,  destination));
    }

    public  String bookFlight(String Flight){
        return lc.reserveSeatXML(Flight);
    }
}



class FlightAdapterJSON implements BookingService{
    protected LegacyFlightSystemJSON lc;
    public FlightAdapterJSON(LegacyFlightSystemJSON LC){
        lc=LC;
    }
    public List<String> searchFlights(String origin, String destination){
        return (lc.getAvailableFlightsJSON( origin,  destination));
    }

    public  String bookFlight(String Flight){
        return lc.confirmBookingJSON(Flight);
    }
}