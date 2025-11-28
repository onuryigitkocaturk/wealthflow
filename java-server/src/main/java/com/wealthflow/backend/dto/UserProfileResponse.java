package com.wealthflow.backend.dto;

import jakarta.validation.constraints.*;

public class UserProfileResponse {

    private Long id;
    private String email;
    private String name;
    private Integer age;
    private Double annualIncome;
    private Integer investmentHorizon;
    private Boolean esgPreference;
    private String riskTolerance;
    private Double riskScore;
    private Boolean riskOverride;
    private String overrideTolerance;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

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

    public Double getRiskScore() {
        return riskScore;
    }

    public void setRiskScore(Double riskScore) {
        this.riskScore = riskScore;
    }

    public Boolean getRiskOverride() {
        return riskOverride;
    }

    public void setRiskOverride(Boolean riskOverride) {
        this.riskOverride = riskOverride;
    }

    public String getOverrideTolerance() {
        return overrideTolerance;
    }

    public void setOverrideTolerance(String overrideTolerance) {
        this.overrideTolerance = overrideTolerance;
    }
}
