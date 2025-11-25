package com.wealthflow.backend.model;

import com.wealthflow.backend.model.enums.RiskTolerance;
import jakarta.persistence.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "user_profiles")
public class UserProfile {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false)
    private String email;

    private String name;

    private Integer age;

    private Double annualIncome;

    private Integer investmentHorizon;

    private Double riskScore;

    private Boolean esgPreference = false;

    @Enumerated(EnumType.STRING)
    private RiskTolerance riskTolerance;

    private Boolean riskOverride;

    @Enumerated(EnumType.STRING)
    private RiskTolerance overrideTolerance;

    @CreationTimestamp
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;

    public UserProfile() {
    }

    public UserProfile(String email, String name, Integer age, Double annualIncome,
                       Integer investmentHorizon, Double riskScore, Boolean esgPreference,
                       RiskTolerance riskTolerance, Boolean riskOverride, RiskTolerance overrideTolerance) {
        this.email = email;
        this.name = name;
        this.age = age;
        this.annualIncome = annualIncome;
        this.investmentHorizon = investmentHorizon;
        this.riskScore = riskScore;
        this.esgPreference = esgPreference;
        this.riskTolerance = riskTolerance;
        this.riskOverride = riskOverride;
        this.overrideTolerance = overrideTolerance;
    }

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

    public Double getRiskScore() {
        return riskScore;
    }

    public void setRiskScore(Double riskScore) {
        this.riskScore = riskScore;
    }

    public Boolean getEsgPreference() {
        return esgPreference;
    }

    public void setEsgPreference(Boolean esgPreference) {
        this.esgPreference = esgPreference;
    }

    public RiskTolerance getRiskTolerance() {
        return riskTolerance;
    }

    public void setRiskTolerance(RiskTolerance riskTolerance) {
        this.riskTolerance = riskTolerance;
    }

    public Boolean getRiskOverride() {
        return riskOverride;
    }

    public void setRiskOverride(Boolean riskOverride) {
        this.riskOverride = riskOverride;
    }

    public RiskTolerance getOverrideTolerance() {
        return overrideTolerance;
    }

    public void setOverrideTolerance(RiskTolerance overrideTolerance) {
        this.overrideTolerance = overrideTolerance;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }
}
