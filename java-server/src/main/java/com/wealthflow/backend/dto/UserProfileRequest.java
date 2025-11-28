package com.wealthflow.backend.dto;

import jakarta.validation.constraints.*;

public class UserProfileRequest {

    @Email(message = "Email is invalid")
    @NotBlank(message = "Email is required")
    private String email;

    @NotBlank(message = "Name is required")
    private String name;

    @NotNull(message = "Age is required")
    @Min(value = 18, message = "Age must be at least 18")
    private Integer age;

    @NotNull(message = "Annual income is required")
    @Positive(message = "Annual income must be positive")
    private Double annualIncome;

    @NotNull(message = "Investment horizon is required")
    @Positive(message = "Investment horizon must be positive")
    private Integer investmentHorizon;

    @NotNull(message = "ESG preference must be provided")
    private Boolean esgPreference;

    @NotNull(message = "Risk tolerance is required")
    private String riskTolerance; // ENUM as String

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getAge() {
        return age;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    public Double getAnnualIncome() {
        return annualIncome;
    }

    public void setAnnualIncome(Double annualIncome) {
        this.annualIncome = annualIncome;
    }

    public Integer getInvestmentHorizon() {
        return investmentHorizon;
    }

    public void setInvestmentHorizon(Integer investmentHorizon) {
        this.investmentHorizon = investmentHorizon;
    }

    public Boolean getEsgPreference() {
        return esgPreference;
    }

    public void setEsgPreference(Boolean esgPreference) {
        this.esgPreference = esgPreference;
    }

    public String getRiskTolerance() {
        return riskTolerance;
    }

    public void setRiskTolerance(String riskTolerance) {
        this.riskTolerance = riskTolerance;
    }
}
