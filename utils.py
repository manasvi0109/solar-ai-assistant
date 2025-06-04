# Solar Panel Technology Specifications
PANEL_TYPES = {
    "monocrystalline": {
        "efficiency": 0.22,  # 22% efficiency
        "cost_per_watt": 45,  # ₹45 per watt
        "area_per_panel": 1.7,  # m²
        "wattage": 400,  # watts per panel
        "lifespan": 25,  # years
        "degradation_rate": 0.005,  # 0.5% per year
        "temperature_coefficient": -0.35,  # %/°C
        "description": "High efficiency, premium option"
    },
    "polycrystalline": {
        "efficiency": 0.18,  # 18% efficiency
        "cost_per_watt": 35,  # ₹35 per watt
        "area_per_panel": 1.8,  # m²
        "wattage": 350,  # watts per panel
        "lifespan": 25,  # years
        "degradation_rate": 0.007,  # 0.7% per year
        "temperature_coefficient": -0.40,  # %/°C
        "description": "Good balance of cost and efficiency"
    },
    "thin_film": {
        "efficiency": 0.12,  # 12% efficiency
        "cost_per_watt": 25,  # ₹25 per watt
        "area_per_panel": 2.5,  # m²
        "wattage": 300,  # watts per panel
        "lifespan": 20,  # years
        "degradation_rate": 0.008,  # 0.8% per year
        "temperature_coefficient": -0.25,  # %/°C
        "description": "Lower cost, flexible installation"
    }
}

# Installation and System Components
SYSTEM_COMPONENTS = {
    "inverter": {
        "string_inverter": {"cost_per_kw": 8000, "efficiency": 0.96, "lifespan": 12},
        "power_optimizer": {"cost_per_kw": 12000, "efficiency": 0.98, "lifespan": 15},
        "microinverter": {"cost_per_kw": 15000, "efficiency": 0.97, "lifespan": 20}
    },
    "mounting": {
        "roof_mounted": {"cost_per_panel": 2500, "tilt_factor": 1.0},
        "ground_mounted": {"cost_per_panel": 4000, "tilt_factor": 1.15},
        "tracking": {"cost_per_panel": 8000, "tilt_factor": 1.35}
    },
    "electrical": {
        "dc_wiring": 1500,  # per kW
        "ac_wiring": 2000,  # per kW
        "monitoring": 5000,  # flat rate
        "safety_equipment": 3000  # per kW
    }
}

# Regional factors for India
REGIONAL_FACTORS = {
    "solar_irradiance": {
        "north": 4.5,  # kWh/m²/day
        "south": 5.5,  # kWh/m²/day
        "east": 4.8,  # kWh/m²/day
        "west": 5.2,  # kWh/m²/day
        "central": 5.0  # kWh/m²/day
    },
    "temperature_factor": 0.85,  # Efficiency reduction due to heat
    "dust_factor": 0.95,  # Efficiency reduction due to dust
    "shading_factor": 0.98  # Efficiency reduction due to partial shading
}

def estimate_panels(usable_area, panel_type="monocrystalline", spacing_factor=1.2):
    """
    Estimate number of solar panels that can fit in the usable area.

    Args:
        usable_area (float): Usable rooftop area in m²
        panel_type (str): Type of solar panel
        spacing_factor (float): Factor for spacing between panels

    Returns:
        dict: Panel count and specifications
    """
    if usable_area <= 0:
        return {"count": 0, "total_capacity": 0, "panel_specs": None}

    panel_specs = PANEL_TYPES.get(panel_type, PANEL_TYPES["monocrystalline"])
    panel_area_with_spacing = panel_specs["area_per_panel"] * spacing_factor

    panel_count = int(usable_area / panel_area_with_spacing)
    total_capacity = panel_count * panel_specs["wattage"] / 1000  # kW

    return {
        "count": max(0, panel_count),
        "total_capacity": total_capacity,
        "panel_specs": panel_specs,
        "panel_type": panel_type
    }


def estimate_installation_cost(panel_data, inverter_type="string_inverter", mounting_type="roof_mounted"):
    """
    Estimate comprehensive installation cost based on panel specifications and system components.

    Args:
        panel_data (dict): Panel information from estimate_panels()
        inverter_type (str): Type of inverter system
        mounting_type (str): Type of mounting system

    Returns:
        dict: Detailed cost breakdown
    """
    if panel_data["count"] <= 0:
        return {"total_cost": 0, "breakdown": {}}

    panel_specs = panel_data["panel_specs"]
    panel_count = panel_data["count"]
    total_capacity = panel_data["total_capacity"]

    # Panel costs
    panel_cost = total_capacity * 1000 * panel_specs["cost_per_watt"]  # Convert kW to W

    # Inverter costs
    inverter_specs = SYSTEM_COMPONENTS["inverter"][inverter_type]
    inverter_cost = total_capacity * inverter_specs["cost_per_kw"]

    # Mounting costs
    mounting_specs = SYSTEM_COMPONENTS["mounting"][mounting_type]
    mounting_cost = panel_count * mounting_specs["cost_per_panel"]

    # Electrical components
    electrical = SYSTEM_COMPONENTS["electrical"]
    electrical_cost = (
        total_capacity * electrical["dc_wiring"] +
        total_capacity * electrical["ac_wiring"] +
        electrical["monitoring"] +
        total_capacity * electrical["safety_equipment"]
    )

    # Installation labor (15% of equipment cost)
    equipment_cost = panel_cost + inverter_cost + mounting_cost + electrical_cost
    labor_cost = equipment_cost * 0.15

    # Permits and inspections (5% of total)
    permits_cost = equipment_cost * 0.05

    # Contingency (10% of total)
    contingency = equipment_cost * 0.10

    total_cost = equipment_cost + labor_cost + permits_cost + contingency

    return {
        "total_cost": int(total_cost),
        "breakdown": {
            "panels": int(panel_cost),
            "inverter": int(inverter_cost),
            "mounting": int(mounting_cost),
            "electrical": int(electrical_cost),
            "labor": int(labor_cost),
            "permits": int(permits_cost),
            "contingency": int(contingency)
        },
        "cost_per_watt": int(total_cost / (total_capacity * 1000)) if total_capacity > 0 else 0
    }


def estimate_savings(panel_data, region="central", electricity_rate=8.5, rate_escalation=0.05):
    """
    Estimate comprehensive savings over system lifetime based on regional factors.

    Args:
        panel_data (dict): Panel information from estimate_panels()
        region (str): Geographic region for solar irradiance
        electricity_rate (float): Current electricity rate in ₹/kWh
        rate_escalation (float): Annual electricity rate increase

    Returns:
        dict: Detailed savings analysis
    """
    if panel_data["count"] <= 0:
        return {"total_savings": 0, "annual_generation": 0, "breakdown": {}}

    panel_specs = panel_data["panel_specs"]
    total_capacity = panel_data["total_capacity"]
    system_lifespan = panel_specs["lifespan"]

    # Regional solar irradiance
    daily_irradiance = REGIONAL_FACTORS["solar_irradiance"][region]
    annual_irradiance = daily_irradiance * 365

    # System efficiency factors
    temp_factor = REGIONAL_FACTORS["temperature_factor"]
    dust_factor = REGIONAL_FACTORS["dust_factor"]
    shading_factor = REGIONAL_FACTORS["shading_factor"]

    # Annual energy generation (first year)
    annual_generation = (
        total_capacity * annual_irradiance *
        temp_factor * dust_factor * shading_factor
    )

    # Calculate savings year by year
    total_savings = 0
    yearly_breakdown = []

    for year in range(system_lifespan):
        # Panel degradation
        degradation_factor = (1 - panel_specs["degradation_rate"]) ** year

        # Electricity rate escalation
        current_rate = electricity_rate * (1 + rate_escalation) ** year

        # Annual generation for this year
        year_generation = annual_generation * degradation_factor

        # Annual savings for this year
        year_savings = year_generation * current_rate
        total_savings += year_savings

        yearly_breakdown.append({
            "year": year + 1,
            "generation": round(year_generation, 2),
            "rate": round(current_rate, 2),
            "savings": round(year_savings, 2)
        })

    # Maintenance costs (reduce from savings)
    annual_maintenance = total_capacity * 1000  # ₹1000 per kW per year
    total_maintenance = annual_maintenance * system_lifespan

    net_savings = total_savings - total_maintenance

    return {
        "total_savings": int(net_savings),
        "gross_savings": int(total_savings),
        "maintenance_costs": int(total_maintenance),
        "annual_generation": round(annual_generation, 2),
        "system_lifespan": system_lifespan,
        "breakdown": {
            "first_year_generation": round(annual_generation, 2),
            "last_year_generation": round(annual_generation * (1 - panel_specs["degradation_rate"]) ** (system_lifespan - 1), 2),
            "average_annual_savings": round(net_savings / system_lifespan, 2),
            "yearly_details": yearly_breakdown[:5]  # First 5 years for display
        }
    }


def calculate_roi(cost_data, savings_data):
    """
    Calculate comprehensive ROI analysis including payback period, NPV, and IRR.

    Args:
        cost_data (dict): Cost information from estimate_installation_cost()
        savings_data (dict): Savings information from estimate_savings()

    Returns:
        dict: Comprehensive ROI analysis
    """
    if cost_data["total_cost"] <= 0 or savings_data["total_savings"] <= 0:
        return {"payback_period": "N/A", "npv": 0, "irr": 0}

    total_cost = cost_data["total_cost"]
    annual_generation = savings_data["annual_generation"]
    system_lifespan = savings_data["system_lifespan"]

    # Simple payback period
    first_year_savings = savings_data["breakdown"]["average_annual_savings"]
    if first_year_savings > 0:
        simple_payback = total_cost / first_year_savings
    else:
        simple_payback = float('inf')

    # Net Present Value (NPV) calculation with 8% discount rate
    discount_rate = 0.08
    npv = -total_cost  # Initial investment

    for year in range(1, system_lifespan + 1):
        # Approximate annual savings (simplified)
        annual_savings = first_year_savings * (0.95 ** (year - 1))  # Degradation factor
        npv += annual_savings / ((1 + discount_rate) ** year)

    # Internal Rate of Return (IRR) - simplified calculation
    # Using approximation method
    irr = calculate_irr_approximation(total_cost, first_year_savings, system_lifespan)

    return {
        "payback_period": round(simple_payback, 1) if simple_payback != float('inf') else "N/A",
        "npv": round(npv, 2),
        "irr": round(irr * 100, 1),  # Convert to percentage
        "cost_per_kwh": round(total_cost / (annual_generation * system_lifespan), 2) if annual_generation > 0 else 0
    }


def calculate_irr_approximation(initial_cost, annual_savings, years):
    """Approximate IRR calculation using iterative method."""
    if annual_savings <= 0:
        return 0

    # Simple approximation: IRR ≈ (annual_savings / initial_cost) - depreciation
    simple_return = annual_savings / initial_cost
    depreciation = 1 / years
    return max(0, simple_return - depreciation)


def get_maintenance_schedule(panel_data):
    """
    Generate maintenance schedule and requirements.

    Args:
        panel_data (dict): Panel information from estimate_panels()

    Returns:
        dict: Maintenance schedule and costs
    """
    if panel_data["count"] <= 0:
        return {"schedule": [], "annual_cost": 0}

    total_capacity = panel_data["total_capacity"]

    maintenance_schedule = [
        {
            "task": "Visual Inspection",
            "frequency": "Monthly",
            "cost_per_year": total_capacity * 200,
            "description": "Check for physical damage, loose connections"
        },
        {
            "task": "Panel Cleaning",
            "frequency": "Quarterly",
            "cost_per_year": total_capacity * 500,
            "description": "Remove dust, debris, and bird droppings"
        },
        {
            "task": "Electrical Testing",
            "frequency": "Annually",
            "cost_per_year": total_capacity * 300,
            "description": "Test electrical connections and performance"
        },
        {
            "task": "Inverter Maintenance",
            "frequency": "Annually",
            "cost_per_year": total_capacity * 400,
            "description": "Check inverter performance and cooling systems"
        }
    ]

    annual_cost = sum(item["cost_per_year"] for item in maintenance_schedule)

    return {
        "schedule": maintenance_schedule,
        "annual_cost": annual_cost,
        "total_25_year_cost": annual_cost * 25
    }


def get_regulatory_requirements():
    """
    Get regulatory requirements and incentives for solar installations in India.

    Returns:
        dict: Regulatory information and incentives
    """
    return {
        "permits_required": [
            "Building Plan Approval",
            "Electrical Safety Clearance",
            "Grid Connection Agreement",
            "Net Metering Application"
        ],
        "building_codes": [
            "IS 16221: Solar PV Installation Code",
            "CEA Technical Standards",
            "State Electricity Board Guidelines",
            "Fire Safety Compliance"
        ],
        "incentives": {
            "central_subsidy": {
                "residential_up_to_3kw": 0.40,  # 40% subsidy
                "residential_3kw_to_10kw": 0.20,  # 20% subsidy
                "description": "Central Financial Assistance"
            },
            "accelerated_depreciation": {
                "rate": 0.40,  # 40% in first year
                "description": "For commercial installations"
            },
            "net_metering": {
                "available": True,
                "description": "Sell excess power back to grid"
            }
        },
        "safety_standards": [
            "Earthing and Lightning Protection",
            "DC and AC Disconnect Switches",
            "Surge Protection Devices",
            "Fire-resistant Cable Routing"
        ]
    }