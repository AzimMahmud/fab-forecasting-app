# Modular Architecture Design: Fabric Consumption Forecasting System

**Date:** 2026-03-08
**Version:** 3.0.0 → 3.1.0 (modular refactor)
**Author:** Azim Mahmud
**Status:** Approved

---

## Executive Summary

This document outlines the refactoring of the Fabric Consumption Forecasting System from a monolithic 3,349-line `app.py` file into a clean, modular three-module architecture. The refactoring improves code organization, maintainability, and testability while maintaining full backward compatibility.

**Goal:** Better code organization through separation of concerns
**Approach:** Three-module architecture (config.py, services.py, app.py)
**Impact:** No breaking changes; entry point and UI remain identical

---

## Architecture Overview

### Module Structure

```
app/
├── config.py          # Configuration & foundational types
├── services.py        # Business logic & services
├── app.py             # UI orchestration & Streamlit interface
├── models/            # ML model files (unchanged)
├── requirements.txt   # Dependencies (unchanged)
└── Dockerfile         # Container config (unchanged)
```

### Design Principles

1. **config.py**: Static configuration, no business logic
2. **services.py**: Pure business logic, no Streamlit dependencies
3. **app.py**: Presentation layer only, imports everything else

### Import Dependencies

```
app.py ──imports──> services.py ──imports──> config.py
                     ↓
              No circular dependencies
```

---

## Component Breakdown

### config.py (~300 lines)

**Purpose:** Centralized configuration and foundational type definitions

**Contents:**
- `AppConfig` class (configuration + environment variables)
- Custom exceptions: `FabricForecastError`, `ModelLoadError`, `ValidationError`, `PredictionError`, `DataLoadError`
- Enums: `ModelType`, `UnitType`, `ProcessingMode`
- Data classes: `PredictionResult`, `OrderInput`, `SystemHealth`
- Constants: `EncodingMaps`
- Logging configuration: `configure_logging()` function

**Dependencies:** None (standalone module)

### services.py (~1,500 lines)

**Purpose:** Business logic and service layer

**Contents:**
- `UnitConverter` class (unit conversion logic)
- `InputValidator` class (validation logic)
- `ModelManager` class (ML model loading and inference)
- `DataGenerator` class (synthetic data generation)
- `ClampedLinearRegression` class (custom regression model)
- `SessionManager` class (session state management)
- `UIHelpers` class (reusable UI utilities)

**Dependencies:** Imports from `config.py` only

### app.py (~1,200 lines)

**Purpose:** UI orchestration and Streamlit interface

**Contents:**
- Streamlit page configuration
- `main()` function (application entry point)
- UI rendering functions (sidebar, tabs, forms)
- Event handlers and callbacks
- Prediction orchestration logic
- File upload handling
- Results display and visualization

**Dependencies:** Imports from both `config.py` and `services.py`

---

## Data Flow

### Single Prediction Request Flow

1. User enters data in `app.py` UI
2. `app.py` validates using `InputValidator` (from services.py)
3. `app.py` calls `ModelManager.predict()` (from services.py)
4. `ModelManager` uses `AppConfig` (from config.py) for model paths
5. `ModelManager` returns `PredictionResult` (from config.py)
6. `app.py` displays results using `UIHelpers` (from services.py)

### Batch Prediction Request Flow

1. User uploads CSV file in `app.py` UI
2. `app.py` validates file using `InputValidator` (from services.py)
3. `app.py` processes rows using `ModelManager` (from services.py)
4. Results aggregated and displayed in `app.py`

---

## Error Handling Strategy

### Exception Hierarchy

```
FabricForecastError (base)
├── ModelLoadError
├── ValidationError
├── PredictionError
└── DataLoadError
```

### Error Handling by Module

**config.py:**
- Defines exception classes only
- No exception handling logic

**services.py:**
- Raises appropriate exceptions for business logic failures
- Examples:
  - `InputValidator.validate()` raises `ValidationError`
  - `ModelManager.load_models()` raises `ModelLoadError`

**app.py:**
- Catches exceptions from services.py
- Displays user-friendly error messages in Streamlit UI
- Logs errors using configured logger

### Error Recovery

- **Validation errors:** Show inline error, allow user to retry
- **Model load errors:** Graceful degradation to fallback models
- **Prediction errors:** Display error message, log details

---

## File Organization & Migration

### Directory Structure After Refactoring

```
app/
├── __init__.py           # New: Package initialization
├── config.py             # New: Configuration module
├── services.py           # New: Business logic module
├── app.py                # Modified: UI orchestration
├── models/               # Unchanged: ML model files
├── logs/                 # Unchanged: Application logs
├── generated_data/       # Unchanged: Synthetic data
├── requirements.txt      # Unchanged: Dependencies
├── Dockerfile            # Unchanged: Container config
├── docker-compose.yml    # Unchanged: Docker compose
└── config.toml           # Unchanged: Streamlit config
```

### Migration Steps

1. Create `config.py` with configuration classes
2. Create `services.py` with business logic classes
3. Update `app.py` to import from new modules
4. Remove moved code from `app.py`
5. Test all functionality to ensure no regressions

### Backward Compatibility

- ✅ Entry point remains `streamlit run app.py`
- ✅ All UI/UX remains identical
- ✅ No API changes for end users
- ✅ Docker configuration unchanged
- ✅ Environment variables unchanged

---

## Module Size Distribution

- **config.py:** ~300 lines (9% of total)
- **services.py:** ~1,500 lines (45% of total)
- **app.py:** ~1,200 lines (36% of total)
- **Shared/imports:** ~349 lines (10% of total)

**Original:** 3,349 lines in single file
**Refactored:** ~3,000 lines across 3 modules (reduced duplication)

---

## Benefits

### Code Organization
- ✅ Clear separation of concerns
- ✅ Easy to navigate and understand
- ✅ Logical grouping by responsibility

### Maintainability
- ✅ Smaller, focused files
- ✅ Reduced cognitive load
- ✅ Easier to locate and fix bugs

### Testability
- ✅ Services can be unit tested independently
- ✅ No Streamlit dependencies in business logic
- ✅ Clear interfaces between modules

### Extensibility
- ✅ Easy to add new services
- ✅ Clear extension points
- ✅ No circular dependencies

---

## Testing Strategy

### Unit Testing
- Test `services.py` classes in isolation
- Mock configuration from `config.py`
- No Streamlit dependencies in service tests

### Integration Testing
- Test `app.py` with real services
- Verify data flow between modules
- Test error handling across boundaries

### Regression Testing
- Verify all existing functionality works
- Compare predictions before/after refactor
- Ensure UI behavior unchanged

---

## Risks & Mitigations

### Risk: Import Errors During Refactoring
**Mitigation:** Create `__init__.py` and test imports incrementally

### Risk: Breaking Changes
**Mitigation:** Maintain backward compatibility; test thoroughly

### Risk: Circular Dependencies
**Mitigation:** Strict linear dependency chain (config → services → app)

---

## Success Criteria

- [ ] All code compiles without errors
- [ ] All existing tests pass
- [ ] Single prediction works identically
- [ ] Batch prediction works identically
- [ ] UI/UX unchanged
- [ ] Docker build succeeds
- [ ] No circular dependencies
- [ ] Documentation updated

---

## Next Steps

1. ✅ Design approved
2. ⏭️ Create detailed implementation plan
3. ⏭️ Execute refactoring
4. ⏭️ Test thoroughly
5. ⏭️ Deploy and monitor

---

**Document Version:** 1.0
**Last Updated:** 2026-03-08
**Status:** Ready for implementation planning
