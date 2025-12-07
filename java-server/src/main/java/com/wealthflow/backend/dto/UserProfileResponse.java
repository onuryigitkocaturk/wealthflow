package com.wealthflow.backend.dto;

import com.wealthflow.backend.model.enums.RiskTolerance;

public record UserProfileResponse(

         Long id,
         String email,
         String name,
         Integer age,
         Double annualIncome,
         Integer investmentHorizon,
         Boolean esgPreference,
         RiskTolerance riskTolerance,
         Double riskScore,
         Boolean riskOverride,
         String overrideTolerance

) {}
